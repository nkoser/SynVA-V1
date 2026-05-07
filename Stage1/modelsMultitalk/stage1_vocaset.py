import torch
import torch.nn as nn
import torch.nn.functional as F

from Stage1.modelsMultitalk.lib.quantizer import VectorQuantizer
from Stage1.modelsMultitalk.lib.base_models import Transformer, LinearEmbedding, PositionalEncoding
from Stage1.base import BaseModel


class VQAutoEncoder(BaseModel):
    """ VQ-GAN model """

    def __init__(self, args):
        super().__init__()
        self.encoder = TransformerEncoder(args)
        self.decoder = TransformerDecoder(args, args.in_dim)
        self.use_factorized = getattr(args, "quantization_mode", "legacy") == "factorized"
        self.factor_proj_mode = getattr(args, "factor_proj", "split")
        self.vq_beta = float(getattr(args, "vq_beta", 0.25))
        self.use_k_head = getattr(args, "use_k_head", False)
        if self.use_k_head:
            self.k_head = nn.Linear(args.hidden_size, args.k_classes)
        if self.use_factorized:
            self.quantizers = nn.ModuleList(
                [VectorQuantizer(args.n_embed, args.factor_dim, beta=self.vq_beta) for _ in range(args.factor_count)]
            )
            if self.factor_proj_mode == "linear_shared":
                self.factor_proj = nn.Linear(args.hidden_size, args.factor_count * args.factor_dim)
            elif self.factor_proj_mode == "linear_per_factor":
                self.factor_proj = nn.ModuleList(
                    [nn.Linear(args.hidden_size, args.factor_dim) for _ in range(args.factor_count)]
                )
            else:
                self.factor_proj = None
        else:
            self.quantize = VectorQuantizer(args.n_embed,
                                            args.zquant_dim,
                                            beta=self.vq_beta)
        self.args = args
        if self.use_factorized:
            expected = args.factor_count * args.factor_dim
            if args.hidden_size != expected:
                raise ValueError(
                    f"hidden_size ({args.hidden_size}) must equal factor_count * factor_dim ({expected})"
                )
        else:
            expected = args.face_quan_num * args.zquant_dim
            if args.hidden_size != expected:
                raise ValueError(
                    f"hidden_size ({args.hidden_size}) must equal face_quan_num * zquant_dim ({expected})"
                )

    def _align_quant_valid_mask(self, quant_valid_mask, target_len, batch_size):
        """Resize [B, L] valid mask to match encoder token length."""
        if quant_valid_mask is None:
            return None
        if quant_valid_mask.dim() != 2:
            raise ValueError(f"quant_valid_mask must have shape [B, L], got {quant_valid_mask.shape}")
        if quant_valid_mask.shape[0] != batch_size:
            raise ValueError(
                f"quant_valid_mask batch size ({quant_valid_mask.shape[0]}) does not match input batch ({batch_size})"
            )
        mask = quant_valid_mask.bool()
        current_len = int(mask.shape[1])
        if current_len == int(target_len):
            return mask
        mask_f = mask.float().unsqueeze(1)
        if current_len > int(target_len):
            # Keep a token valid if any source position in its pooled bin is valid.
            resized = F.adaptive_max_pool1d(mask_f, output_size=int(target_len))
        else:
            resized = F.interpolate(mask_f, size=int(target_len), mode="nearest")
        return resized.squeeze(1) > 0.5



    def encode(self, x, x_a=None, attn_mask=None, quant_valid_mask=None): 
        h = self.encoder(x, attn_mask=attn_mask) ## x --> z'
        if self.use_factorized:
            if self.factor_proj_mode == "linear_shared":
                h = self.factor_proj(h)
                h = h.view(x.shape[0], -1, self.args.factor_count, self.args.factor_dim)
            elif self.factor_proj_mode == "linear_per_factor":
                h = torch.stack([proj(h) for proj in self.factor_proj], dim=2)
            else:
                h = h.view(x.shape[0], -1, self.args.factor_count, self.args.factor_dim)
            quant_mask = self._align_quant_valid_mask(
                quant_valid_mask,
                target_len=h.shape[1],
                batch_size=x.shape[0],
            )
            quant_list = []
            loss_list = []
            info_list = []
            for i in range(self.args.factor_count):
                q, loss, info = self.quantizers[i](h[:, :, i, :], valid_mask=quant_mask)
                quant_list.append(q)
                loss_list.append(loss)
                info_list.append(info)
            quant = torch.cat(quant_list, dim=1)
            emb_loss = torch.stack(loss_list).mean()
            info = info_list
            return quant, emb_loss, info
        h = h.view(x.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim)
        flat_quant_tokens = int(h.shape[1] * self.args.face_quan_num)
        if quant_valid_mask is not None and quant_valid_mask.dim() == 2 and quant_valid_mask.shape[1] == flat_quant_tokens:
            quant_mask = quant_valid_mask.bool()
        else:
            quant_mask = self._align_quant_valid_mask(
                quant_valid_mask,
                target_len=h.shape[1],
                batch_size=x.shape[0],
            )
            if quant_mask is not None:
                quant_mask = quant_mask.repeat_interleave(self.args.face_quan_num, dim=1)
        h = h.view(x.shape[0], -1, self.args.zquant_dim)
        quant, emb_loss, info = self.quantize(h, valid_mask=quant_mask) ## finds nearest quantization
        
        return quant, emb_loss, info
        #return h


    def decode2(self, quant):
        #BCL
        quant = quant.permute(0,2,1)
        #quant = quant.view(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        #quant = quant.reshape(quant.shape[0], -1, self.args.face_quan_num, self.args.zquant_dim).contiguous()
        quant = quant.reshape(quant.shape[0], self.args.zquant_dim, self.args.face_quan_num).contiguous()
        quant = quant.view(quant.shape[0], -1,  self.args.face_quan_num*self.args.zquant_dim).contiguous()
        quant = quant.permute(0,2,1).contiguous()
        dec = self.decoder(quant) ## z' --> x
        return dec
    
    def decode(self, quant, return_features=False, attn_mask=None):
        if self.use_factorized:
            if quant.dim() == 3 and quant.shape[1] != self.args.hidden_size and quant.shape[2] == self.args.hidden_size:
                quant = quant.permute(0, 2, 1).contiguous()
            return self.decoder(quant, attn_mask=attn_mask) if not return_features else self.decoder(quant, return_features=True, attn_mask=attn_mask)
        # Assuming quant has the shape (batch_size, num_tokens, zquant_dim)
        # Step 1: Reshape or permute the input tensor as required by your model
        # Change the shape to (B, num_tokens, zquant_dim) if needed
        quant = quant.permute(0, 2, 1)  # Change shape to (batch_size, zquant_dim, num_tokens)

        # Step 2: Reshape for the decoder input
        quant = quant.reshape(quant.shape[0], -1, self.args.face_quan_num * self.args.zquant_dim)

        # Step 3: Ensure the shape matches the expected input of the decoder
        # The decoder might expect the shape to be (batch_size, num_features, seq_len)
        quant = quant.permute(0, 2, 1)  # Change back to (batch_size, features, seq_len)

        # Step 4: Pass through the decoder
        if return_features:
            dec, dec_feat = self.decoder(quant, return_features=True, attn_mask=attn_mask)  # z' --> x
            return dec, dec_feat
        dec = self.decoder(quant, attn_mask=attn_mask)  # z' --> x

        return dec

    def forward(self, x, attn_mask=None, quant_valid_mask=None):
        #template = template.unsqueeze(1) #B,V*3 -> B, 1, V*3
        #x = x - template

        ###x.shape: [B, L C]
        # Assuming your model has named layers

        quant, emb_loss, info = self.encode(x, attn_mask=attn_mask, quant_valid_mask=quant_valid_mask)
        #quant = self.encode(x)
        ### quant [B, C, L]
        if self.use_k_head:
            dec, dec_feat = self.decode(quant, return_features=True, attn_mask=attn_mask)
            k_logits = self.k_head(dec_feat)
            return dec, emb_loss, info, k_logits
        dec = self.decode(quant, attn_mask=attn_mask)
        #dec = dec + template
        return dec, emb_loss, info
        #return dec


    def sample_step(self, x, x_a=None):
        
        quant_z, _, info = self.encode(x, x_a)
        
        x_sample_det = self.decode(quant_z)
        if self.use_factorized:
            btc = quant_z.shape[0], quant_z.shape[2], self.args.factor_dim
            indices = [item[2] for item in info]
        else:
            btc = quant_z.shape[0], quant_z.shape[2], quant_z.shape[1]
            indices = info[2]
        x_sample_check = self.decode_to_img(indices, btc)
        return x_sample_det, x_sample_check
    
    def sample_step_wc(self, x, x_a=None):
        quant_z, _, info = self.encode(x, x_a)
        x_sample_det = self.decode(quant_z)
        
        return x_sample_det
    
    @torch.no_grad()
    def get_quant_no_grad(self, x, x_a=None):

        quant_z, _, info = self.encode(x, x_a)
        indices = [item[2] for item in info] if self.use_factorized else info[2]
        return quant_z, indices
    
    def get_quant(self, x, x_a=None):

        quant_z, _, info = self.encode(x, x_a)
        indices = [item[2] for item in info] if self.use_factorized else info[2]
        return quant_z, indices

    def get_distances(self, x):
        h = self.encoder(x) ## x --> z'
        d = self.quantize.get_distance(h)
        return d

    def get_quant_from_d(self, d, btc):
        min_encoding_indices = torch.argmin(d, dim=1).unsqueeze(1)
        x = self.decode_to_img(min_encoding_indices, btc)
        return x

    @torch.no_grad()
    def entry_to_feature(self, index, zshape):
        if self.use_factorized:
            if not isinstance(index, (list, tuple)):
                raise ValueError("factorized mode expects a list of indices")
            quant_list = []
            for i, idx in enumerate(index):
                idx = idx.long()
                quant_i = self.quantizers[i].get_codebook_entry(idx.reshape(-1), shape=None)
                quant_i = torch.reshape(quant_i, zshape)
                quant_list.append(quant_i)
            quant = torch.cat(quant_list, dim=2)
            return quant.permute(0, 2, 1).contiguous()
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape)
        return quant_z



    @torch.no_grad()
    def decode_to_img(self, index, zshape):
        if self.use_factorized:
            if not isinstance(index, (list, tuple)):
                raise ValueError("factorized mode expects a list of indices")
            quant_list = []
            for i, idx in enumerate(index):
                idx = idx.long()
                quant_i = self.quantizers[i].get_codebook_entry(idx.reshape(-1), shape=None)
                quant_i = torch.reshape(quant_i, zshape).permute(0, 2, 1)
                quant_list.append(quant_i)
            quant_z = torch.cat(quant_list, dim=1)
            x = self.decode(quant_z)
            return x
        index = index.long()
        quant_z = self.quantize.get_codebook_entry(index.reshape(-1),
                                                   shape=None)
        quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        x = self.decode(quant_z)
        return x
    
    @torch.no_grad()
    def decode_quant(self, quant_z, zshape):
        #quant_z = torch.reshape(quant_z, zshape).permute(0,2,1) # B L 1 -> B L C -> B C L
        #quant_z = quant_z.permute(0,2,1)
        #print("quant_z", quant_z.shape)
        x = self.decode(quant_z)
        return x

    @torch.no_grad()
    def decode_logit(self, logits, zshape):
        if logits.dim() == 3:

            probs = F.softmax(logits, dim=-1)
            _, ix = torch.topk(probs, k=1, dim=-1)

        else:
            ix = logits
        #ix = torch.reshape(ix, (-1,1))
 
        x = self.decode_to_img(ix, zshape)
        return x

    def get_logit(self, logits, sample=True, filter_value=-float('Inf'),
                  temperature=0.7, top_p=0.9, sample_idx=None):
        """ function that samples the distribution of logits. (used in test)
        if sample_idx is None, we perform nucleus sampling
        """
        logits = logits / temperature
        sample_idx = 0
        ##########
        probs = F.softmax(logits, dim=-1) # B, N, embed_num
        if sample:
            ## multinomial sampling
            shape = probs.shape
            probs = probs.reshape(shape[0]*shape[1],shape[2])
            ix = torch.multinomial(probs, num_samples=sample_idx+1)
            probs = probs.reshape(shape[0],shape[1],shape[2])
            ix = ix.reshape(shape[0],shape[1])
        else:
            ## top 1; no sampling
            _, ix = torch.topk(probs, k=1, dim=-1)
        return ix, probs


class TransformerEncoder(nn.Module):
  """ Encoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args):
    super().__init__()
    self.args = args
    size = self.args.in_dim
    dim = self.args.hidden_size
    self.vertice_mapping = nn.Sequential(nn.Linear(size,dim), nn.LeakyReLU(self.args.neg, True))
    if args.quant_factor == 0:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
    else:
        layers = [nn.Sequential(
                    nn.Conv1d(dim,dim,5,stride=2,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                    )]
        for _ in range(1, args.quant_factor):
            layers += [nn.Sequential(
                        nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                    padding_mode='replicate'),
                        nn.LeakyReLU(self.args.neg, True),
                        nn.InstanceNorm1d(dim, affine=args.INaffine),
                        nn.MaxPool1d(2)
                        )]
    self.squasher = nn.Sequential(*layers)
    dropout = getattr(self.args, 'dropout', 0.1)
    self.encoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
                self.args.num_hidden_layers,
        num_attention_heads=\
                self.args.num_attention_heads,
        intermediate_size=\
                self.args.intermediate_size,
        dropout=dropout)
    self.encoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.encoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

  def forward(self, inputs, return_features=False, attn_mask=None):
    ## downdample into path-wise length seq before passing into transformer
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            mask_4d = attn_mask
        elif attn_mask.dim() == 3:
            mask_4d = attn_mask.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
        if self.args.quant_factor > 0:
            # Downsample the attention mask to match reduced sequence length
            # Reduction factor: 2^quant_factor
            reduction = 2 ** self.args.quant_factor
            # Use max pooling to downsample: if ANY position in block is valid (0), keep it valid
            # attn_mask uses -inf for masked, 0 for valid
            # Reshape to [B, 1, L, L] for pooling
            # Pool both dimensions
            mask_4d = F.max_pool2d(mask_4d, kernel_size=reduction, stride=reduction)
        mask_info = {'max_mask': mask_4d.shape[-1], 'mask_index': -1, 'mask': mask_4d}
    else:
        mask_info = {'max_mask': None, 'mask_index': -1, 'mask': None}
    #print("inputs", inputs.device)
    inputs = self.vertice_mapping(inputs)
    #print(f"After vertice_mapping: {inputs}")
    for param in self.vertice_mapping.parameters():
        if torch.isnan(param).any():
            print("NaN detected in model vertice mapping parameters!")
    inputs = self.squasher(inputs.permute(0,2,1)).permute(0,2,1) # [N L C]
    #print(f"After squasher: {inputs}")
    for param in self.squasher.parameters():
        if torch.isnan(param).any():
            print("NaN detected in model squasher parameters!")
    encoder_features = self.encoder_linear_embedding(inputs)
    #print(f"After encoder_linear_embedding: {encoder_features}")

    encoder_features = self.encoder_pos_embedding(encoder_features)
    #print(f"After encoder_pos_embedding: {encoder_features}")

    encoder_features = self.encoder_transformer((encoder_features, mask_info))
    #print(f"After encoder_transformer: {encoder_features}")

    return encoder_features


class TransformerDecoder(nn.Module):
  """ Decoder class for VQ-VAE with Transformer backbone """

  def __init__(self, args, out_dim, is_audio=False):
    super().__init__()
    self.args = args
    size=self.args.hidden_size
    dim=self.args.hidden_size
    self.expander = nn.ModuleList()
    if args.quant_factor == 0:
        self.expander.append(nn.Sequential(
                    nn.Conv1d(size,dim,5,stride=1,padding=2,
                                padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))
    else:
        self.expander.append(nn.Sequential(
                    nn.ConvTranspose1d(size,dim,5,stride=2,padding=2,
                                        output_padding=1,
                                        padding_mode='replicate'),
                    nn.LeakyReLU(self.args.neg, True),
                    nn.InstanceNorm1d(dim, affine=args.INaffine)
                            ))                      
        num_layers = args.quant_factor+2 \
            if is_audio else args.quant_factor

        for _ in range(1, num_layers):
            self.expander.append(nn.Sequential(
                                nn.Conv1d(dim,dim,5,stride=1,padding=2,
                                        padding_mode='replicate'),
                                nn.LeakyReLU(self.args.neg, True),
                                nn.InstanceNorm1d(dim, affine=args.INaffine),
                                ))
    dropout = getattr(self.args, 'dropout', 0.1)
    self.decoder_transformer = Transformer(
        in_size=self.args.hidden_size,
        hidden_size=self.args.hidden_size,
        num_hidden_layers=\
            self.args.num_hidden_layers,
        num_attention_heads=\
            self.args.num_attention_heads,
        intermediate_size=\
            self.args.intermediate_size,
        dropout=dropout)
    self.decoder_pos_embedding = PositionalEncoding(
        self.args.hidden_size)
    self.decoder_linear_embedding = LinearEmbedding(
        self.args.hidden_size,
        self.args.hidden_size)

    self.vertice_map_reverse = nn.Linear(args.hidden_size,out_dim)

  def forward(self, inputs, return_features=False, attn_mask=None):
    # Compute output sequence length after expander
    L_in = inputs.shape[2]  # [B, C, L]
    if self.args.quant_factor > 0:
        # First ConvTranspose doubles, then repeat_interleave doubles for each subsequent layer
        L_out = L_in * 2  # ConvTranspose
        for _ in range(1, self.args.quant_factor):
            L_out = L_out * 2  # repeat_interleave
    else:
        L_out = L_in
    
    if attn_mask is not None:
        if attn_mask.dim() == 4:
            mask_4d = attn_mask
        elif attn_mask.dim() == 3:
            mask_4d = attn_mask.unsqueeze(1)
        else:
            raise ValueError(f"Unsupported attn_mask shape: {attn_mask.shape}")
        if self.args.quant_factor > 0:
            # Upsample the attention mask to match expanded sequence length
            # If mask is already at target length, keep it unchanged.
            if mask_4d.shape[-1] != L_out:
                mask_4d = F.interpolate(mask_4d, size=(L_out, L_out), mode='nearest')
        mask_info = {'max_mask': mask_4d.shape[-1], 'mask_index': -1, 'mask': mask_4d}
    else:
        mask_info = {'max_mask': None, 'mask_index': -1, 'mask': None}
    
    ## upsample into original length seq before passing into transformer
    for i, module in enumerate(self.expander):
        inputs = module(inputs)
        if i > 0:
            inputs = inputs.repeat_interleave(2, dim=2)
    inputs = inputs.permute(0,2,1) #BLC
    decoder_features = self.decoder_linear_embedding(inputs)
    decoder_features = self.decoder_pos_embedding(decoder_features)

    decoder_features = self.decoder_transformer((decoder_features, mask_info))
    pred_recon = self.vertice_map_reverse(decoder_features)
    if return_features:
        return pred_recon, decoder_features
    return pred_recon
  
def validate(val_loader, model, loss_fn, epoch, cfg, device):
    accumulated_loss = 0
    accumulated_rec = 0
    accumulated_quant = 0
    model.eval()

    with torch.no_grad():
        for inputs in val_loader:
            
            inputs = inputs.to(device)
            out, quant_loss, info = model(inputs)
            
            # LOSS
            loss, loss_details = loss_fn(out, inputs, quant_loss, quant_loss_weight=cfg.quant_loss_weight)
            accumulated_loss += loss
            accumulated_rec += loss_details[0].item()
            accumulated_quant += loss_details[1].item()
        

        # Calculate the average loss over the entire dataset
        avg_loss = accumulated_loss / len(val_loader)
        rec_loss = accumulated_rec / len(val_loader)
        quant_loss = accumulated_quant / len(val_loader)

            

    return rec_loss, quant_loss
