import glob
import os
import pickle
import shutil
import subprocess
import sys
import traceback
from os.path import join

import numpy as np
import networkx as nx
import vtk
from vtk.util.numpy_support import vtk_to_numpy

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

from Preprocessing_v2.ResamplingRDP import (
    interpolarRDP_conRadio,
    resample_centerline_step,
    resample_centerline_minimal_adaptive,
    resample_centerline_event,
    resample_centerline_vmtk,
    vtpToObj,
)
from Preprocessing_v2.splines import (
    calculate_splines,
    limpiarRadiosSplines,
    traversefeaturesSerializado,
    binarizar,
    grafo2arbol,
)
from Preprocessing_v2.parseObj import calcularMatriz, calcularMatrizSplines
import Preprocessing_v2.Arbol as modelo


_VMTK_EXTRACTOR_CODE = (
    "import sys\n"
    "from vmtk import pypes\n"
    "script, input_path, output_path = sys.argv[1:4]\n"
    "input_file = f'-ifile \"{input_path}\"'\n"
    "output_file = f'-ofile \"{output_path}\"'\n"
    "pypes.PypeRun(script + ' ' + input_file + ' ' + output_file)\n"
)


def _run_vmtk_network_extraction(script, input_path, output_path, timeout_s):
    timeout = None
    if timeout_s is not None and float(timeout_s) > 0:
        timeout = float(timeout_s)

    cmd = [sys.executable, "-c", _VMTK_EXTRACTOR_CODE, script, input_path, output_path]

    try:
        result = subprocess.run(cmd, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"Timed out after {timeout_s}s"
    except Exception as exc:
        return False, f"Failed to launch vmtk subprocess: {exc}"

    if result.returncode != 0:
        return False, f"vmtk exited with code {result.returncode}"
    if not os.path.exists(output_path):
        return False, "Output file was not created"
    if os.path.getsize(output_path) == 0:
        return False, "Output file is empty"
    return True, ""


def _inject_advancementratio(script, ratio):
    parts = script.split()
    if "-advancementratio" in parts:
        idx = parts.index("-advancementratio")
        if idx + 1 < len(parts):
            parts[idx + 1] = str(ratio)
        else:
            parts.append(str(ratio))
        return " ".join(parts)
    return f"{script} -advancementratio {ratio}"


def _read_vtp_polydata(path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(path)
    reader.Update()
    polydata = vtk.vtkPolyData()
    polydata.DeepCopy(reader.GetOutput())
    return polydata


def _is_empty_polydata(polydata):
    if polydata is None:
        return True
    return polydata.GetNumberOfPoints() <= 0 or polydata.GetNumberOfCells() <= 0


class PreprocessingPipeline:
    STEP_ORDER = [
        "copy_raw_data",
        "normalize_meshes",
        "extract_centerlines",
        "resample_centerlines",
        "centerlines_to_obj",
        "vessels_to_obj",
        "radius_arrays",
        "splines",
        "graphs",
        "trees",
        "graphs_splines",
        "trees_splines",
    ]

    REQUIRED_PATHS = [
        "raw_meshes",
        "vessels_normalized",
        "centerlines",
        "centerlines_resampled",
        "centerlines_resampled_obj",
        "vessels_obj",
        "radius_arrays",
        "splines_coef",
        "splines_knots",
        "grafos",
        "trees_numpy",
        "trees_serialized",
        "grafos_splines",
        "trees_splines",
    ]

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = cfg.get("_paths") or {}
        self.params = cfg.get("params", {})
        self.flags = cfg.get("flags", {})

    @staticmethod
    def _ensure_dir(path):
        os.makedirs(path, exist_ok=True)

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return []
            if "," in text:
                return [part.strip() for part in text.split(",") if part.strip()]
            return [text]
        if isinstance(value, (list, tuple, set)):
            out = []
            for item in value:
                text = str(item).strip()
                if text:
                    out.append(text)
            return out
        text = str(value).strip()
        return [text] if text else []

    @staticmethod
    def _case_id_from_mesh_path(mesh_path):
        parts = os.path.normpath(mesh_path).split(os.sep)
        if len(parts) >= 3:
            return parts[-3]
        return os.path.splitext(os.path.basename(mesh_path))[0]

    @staticmethod
    def _read_polydata(mesh_path):
        ext = os.path.splitext(mesh_path)[1].lower()
        if ext == ".vtp":
            reader = vtk.vtkXMLPolyDataReader()
        elif ext == ".obj":
            reader = vtk.vtkOBJReader()
        else:
            raise ValueError(f"Unsupported mesh extension: {ext}")

        reader.SetFileName(mesh_path)
        reader.Update()
        polydata = vtk.vtkPolyData()
        polydata.DeepCopy(reader.GetOutput())
        return polydata

    def _discover_raw_meshes(self):
        input_dir = self.paths["raw_meshes"]
        patterns = self._as_list(self.params.get("raw_mesh_patterns"))
        if not patterns:
            patterns = ["*_models/*/surface/*.vtp"]

        dataset_sep = str(self.params.get("raw_mesh_dataset_separator", "_"))
        dataset_filter = {
            item.lower() for item in self._as_list(self.params.get("raw_mesh_datasets"))
        }

        by_case_id = {}
        for pattern in patterns:
            for mesh_path in glob.glob(join(input_dir, pattern)):
                if not os.path.isfile(mesh_path):
                    continue

                ext = os.path.splitext(mesh_path)[1].lower()
                if ext not in {".vtp", ".obj"}:
                    continue

                case_id = self._case_id_from_mesh_path(mesh_path)
                dataset_name = case_id.split(dataset_sep, 1)[0] if dataset_sep else case_id
                if dataset_filter and dataset_name.lower() not in dataset_filter:
                    continue

                by_case_id.setdefault(case_id, mesh_path)

        return sorted(by_case_id.items(), key=lambda x: x[0])

    @staticmethod
    def _resolve_paths(cfg, config_dir):
        paths = cfg.get("paths", {})
        resolved = {}
        for key, value in paths.items():
            if value is None:
                resolved[key] = None
                continue
            if os.path.isabs(value):
                resolved[key] = value
            else:
                resolved[key] = os.path.normpath(os.path.join(config_dir, value))
        return resolved

    @staticmethod
    def _apply_path_defaults(paths):
        raw_root = paths.get("raw_root")
        output_root = paths.get("output_root")

        if raw_root and not paths.get("raw_meshes"):
            paths["raw_meshes"] = raw_root

        if output_root:
            defaults = {
                "vessels_normalized": os.path.join(output_root, "vesselsNormalized"),
                "centerlines": os.path.join(output_root, "centerlines"),
                "centerlines_resampled": os.path.join(output_root, "centerlinesResampled"),
                "centerlines_resampled_obj": os.path.join(output_root, "centerlinesResampledOBJ"),
                "vessels_obj": os.path.join(output_root, "vesselsOBJ"),
                "radius_arrays": os.path.join(output_root, "radius_arrays"),
                "splines_coef": os.path.join(output_root, "splines", "coeficientes"),
                "splines_knots": os.path.join(output_root, "splines", "knots"),
                "grafos": os.path.join(output_root, "grafos"),
                "trees_numpy": os.path.join(output_root, "TreesNumpy"),
                "trees_serialized": os.path.join(output_root, "Trees"),
                "grafos_splines": os.path.join(output_root, "grafosSplines"), 
                "trees_splines": os.path.join(output_root, "TreesSplines"),
            }
            for key, value in defaults.items():
                paths.setdefault(key, value)

        return paths

    @staticmethod
    def _ensure_trailing_sep(path):
        if path.endswith(os.sep):
            return path
        return path + os.sep

    @classmethod
    def load_config(cls, path):
        config_path = os.path.abspath(path)
        config_dir = os.path.dirname(config_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        cfg = cfg or {}
        cfg["_config_path"] = config_path
        cfg["_config_dir"] = config_dir
        cfg["paths"] = cls._apply_path_defaults(cfg.get("paths", {}) or {})
        cfg["_paths"] = cls._resolve_paths(cfg, config_dir)
        return cfg

    def _validate_paths(self):
        for key in self.REQUIRED_PATHS:
            if key not in self.paths:
                raise ValueError("Missing path in config: " + key)

    def normalize_meshes(self):
        self._ensure_dir(self.paths["vessels_normalized"])
        overwrite = bool(self.params.get("normalize_overwrite", False))
        writer = vtk.vtkXMLPolyDataWriter()
        meshes = self._discover_raw_meshes()
        print("Found raw meshes for normalization:", len(meshes))

        for case_id, src_path in meshes:
            dst_path = join(self.paths["vessels_normalized"], case_id + ".vtp")
            if os.path.exists(dst_path) and not overwrite:
                continue

            try:
                polydata = self._read_polydata(src_path)
            except Exception:
                print("Could not read mesh:", src_path)
                traceback.print_exc()
                continue

            # Merge duplicate vertices so the surface is topologically connected
            # (raw OBJ exports often have one vertex per triangle corner, which
            # breaks vmtknetworkextraction). Tolerance 0 = bit-exact merging.
            try:
                cleaner = vtk.vtkCleanPolyData()
                cleaner.SetInputData(polydata)
                cleaner.SetTolerance(0.0)
                cleaner.PointMergingOn()
                cleaner.ConvertLinesToPointsOff()
                cleaner.ConvertPolysToLinesOff()
                cleaner.ConvertStripsToPolysOff()
                cleaner.Update()
                cleaned = vtk.vtkPolyData()
                cleaned.DeepCopy(cleaner.GetOutput())
                if cleaned.GetNumberOfPoints() > 0 and cleaned.GetNumberOfCells() > 0:
                    polydata = cleaned
            except Exception:
                print("vtkCleanPolyData failed, using raw mesh:", src_path)
                traceback.print_exc()

            points = polydata.GetPoints()
            if points is None:
                print("No points found in:", src_path)
                continue

            n_points = points.GetNumberOfPoints()
            coords = np.array([points.GetPoint(i) for i in range(n_points)])

            mins = coords.min(axis=0)
            maxs = coords.max(axis=0)
            center = (maxs + mins) / 2
            half_ranges = (maxs - mins) / 2
            uniform_scale = np.max(half_ranges)
            if uniform_scale == 0:
                uniform_scale = 1.0

            normalized_coords = (coords - center) / uniform_scale

            new_points = vtk.vtkPoints()
            for p in normalized_coords:
                new_points.InsertNextPoint(p.tolist())

            polydata.SetPoints(new_points)
            polydata.Modified()

            writer.SetFileName(dst_path)
            writer.SetInputData(polydata)
            writer.Write()
            print("Normalized:", src_path)

    def extract_centerlines(self):
        self._ensure_dir(self.paths["centerlines"])
        vessels = [f for f in os.listdir(self.paths["vessels_normalized"]) if f.endswith(".vtp")]
        existing = set(os.listdir(self.paths["centerlines"]))
        vmtk_script = (self.params.get("vmtk_script", "") or "").strip()
        if not vmtk_script:
            raise ValueError("params.vmtk_script must not be empty")

        timeout_s = self.params.get("extract_centerlines_timeout_s", 600)
        try:
            timeout_s = float(timeout_s)
        except Exception:
            timeout_s = 600.0
        if timeout_s < 0:
            timeout_s = 0.0

        fallback_ratios = []
        for value in self._as_list(
            self.params.get("extract_centerlines_fallback_ratios", [1.02, 1.01, 1.005])
        ):
            try:
                parsed = float(value)
            except Exception:
                continue
            if parsed > 0:
                fallback_ratios.append(parsed)

        script_candidates = [vmtk_script]
        script_candidates.extend(_inject_advancementratio(vmtk_script, ratio) for ratio in fallback_ratios)
        script_candidates.extend(
            s.strip() for s in self._as_list(self.params.get("extract_centerlines_fallback_scripts")) if s.strip()
        )

        # Preserve order while removing duplicates.
        deduped = []
        seen = set()
        for script in script_candidates:
            if script in seen:
                continue
            seen.add(script)
            deduped.append(script)
        script_candidates = deduped

        failures = []

        for file in vessels:
            base = os.path.splitext(file)[0]
            out_name = base + "-network.vtp"
            if out_name in existing:
                existing_path = os.path.join(self.paths["centerlines"], out_name)
                try:
                    existing_poly = _read_vtp_polydata(existing_path)
                except Exception:
                    existing_poly = None
                if not _is_empty_polydata(existing_poly):
                    continue
                print("Existing centerline is empty, regenerating:", out_name)
            print("Processing file:", file)
            input_path = os.path.join(self.paths["vessels_normalized"], file)
            output_path = os.path.join(self.paths["centerlines"], out_name)
            success = False
            last_error = "unknown error"

            for idx, script in enumerate(script_candidates, start=1):
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass

                print(f"  Attempt {idx}/{len(script_candidates)}: {script}")
                ok, error = _run_vmtk_network_extraction(
                    script=script,
                    input_path=input_path,
                    output_path=output_path,
                    timeout_s=timeout_s,
                )
                if ok:
                    try:
                        extracted = _read_vtp_polydata(output_path)
                    except Exception as exc:
                        extracted = None
                        error = f"Could not read vmtk output: {exc}"
                    if not _is_empty_polydata(extracted):
                        success = True
                        print("  Extraction finished:", out_name)
                        break
                    error = "vmtk output contains no points/cells"
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass

                last_error = error
                print("  Extraction failed:", error)

            if not success:
                failures.append((file, last_error))
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                print("Skipping file after failed retries:", file)

        if failures:
            print("extract_centerlines failures:", len(failures))
            for file, error in failures:
                print(" -", file, "|", error)

    def resample_centerlines(self):
        self._ensure_dir(self.paths["centerlines_resampled"])
        reader = vtk.vtkXMLPolyDataReader()
        writer = vtk.vtkXMLPolyDataWriter()
        method = (self.params.get("resample_method", "rdp") or "rdp").lower()
        epsilon = self.params.get("resample_epsilon", 0.02)
        step = self.params.get("resample_step", 0.003)
        use_radius = bool(self.params.get("resample_use_radius", False))
        step_min = self.params.get("resample_step_min", 0.001)
        step_max = self.params.get("resample_step_max", 0.006)
        base_step = self.params.get("resample_base_step")
        radius_scale = self.params.get("resample_radius_scale", 0.3)
        radius_mode = self.params.get("resample_radius_mode", "inverse")
        curv_threshold = self.params.get("resample_curv_threshold", 0.0)
        curv_boost = self.params.get("resample_curv_boost", 0.0)
        drds_threshold = self.params.get("resample_drds_threshold", 0.0)
        drds_boost = self.params.get("resample_drds_boost", 1.0)
        junction_window = self.params.get("resample_junction_window", 0.0)
        junction_factor = self.params.get("resample_junction_factor", 1.0)
        junction_degree = self.params.get("resample_junction_degree", 3)
        max_points_per_branch = self.params.get("resample_max_points_per_branch")
        event_step = self.params.get("resample_event_step", 0.005)
        event_window = self.params.get("resample_event_window", 0.02)
        geom_tol = self.params.get("resample_geom_tol", 0.02)
        rad_tol = self.params.get("resample_rad_tol", 0.04)
        w_rad = self.params.get("resample_w_rad", 1.0)
        clean_tol = self.params.get("resample_clean_tol", 0.0)
        minimal_prune_passes = self.params.get("resample_minimal_prune_passes", 4)
        junction_keep_k = self.params.get("resample_junction_keep_k", 0)
        junction_keep_window = self.params.get("resample_junction_keep_window", 0.0)

        centerlines = os.listdir(self.paths["centerlines"])
        for file in centerlines:
            if not file.endswith(".vtp"):
                continue
            output_path = os.path.join(self.paths["centerlines_resampled"], file)
            if os.path.exists(output_path):
                try:
                    existing_resampled = _read_vtp_polydata(output_path)
                except Exception:
                    existing_resampled = None
                if not _is_empty_polydata(existing_resampled):
                    continue
                try:
                    os.remove(output_path)
                except OSError:
                    pass
            print("Resampling:", file)
            reader.SetFileName(os.path.join(self.paths["centerlines"], file))
            reader.Update()
            centerline = reader.GetOutput()
            if _is_empty_polydata(centerline):
                print("Skipping empty centerline:", file)
                continue
            if method == "rdp":
                resampled = interpolarRDP_conRadio(centerline, epsilon)
            elif method == "step":
                resampled = resample_centerline_step(
                    centerline,
                    step=step,
                    use_radius=use_radius,
                    step_min=step_min,
                    step_max=step_max,
                    base_step=base_step,
                    radius_scale=radius_scale,
                    radius_mode=radius_mode,
                    curv_threshold=curv_threshold,
                    curv_boost=curv_boost,
                    drds_threshold=drds_threshold,
                    drds_boost=drds_boost,
                    junction_window=junction_window,
                    junction_factor=junction_factor,
                    junction_degree=junction_degree,
                    max_points_per_branch=max_points_per_branch,
                )
            elif method == "event":
                resampled = resample_centerline_event(
                    centerline,
                    base_step=base_step if base_step is not None else step,
                    event_step=event_step,
                    event_window=event_window,
                    drds_threshold=drds_threshold,
                    curv_threshold=curv_threshold,
                    junction_window=junction_window,
                    junction_degree=junction_degree,
                )
            elif method == "minimal":
                resampled = resample_centerline_minimal_adaptive(
                    centerline,
                    geom_tol=geom_tol,
                    rad_tol=rad_tol,
                    w_rad=w_rad,
                    junction_degree=junction_degree,
                    junction_keep_k=junction_keep_k,
                    junction_keep_window=junction_keep_window,
                    max_points_per_branch=max_points_per_branch,
                    clean_tol=clean_tol,
                    minimal_prune_passes=minimal_prune_passes,
                )
            elif method == "vmtk":
                resampled = resample_centerline_vmtk(centerline, step=step)
            else:
                raise ValueError(f"Unknown resample_method: {method}")
            if _is_empty_polydata(resampled):
                print("Skipping empty resampled centerline:", file)
                continue
            writer.SetFileName(output_path)
            writer.SetInputData(resampled)
            writer.Write()

    def centerlines_to_obj(self):
        self._ensure_dir(self.paths["centerlines_resampled_obj"])
        centerlines = os.listdir(self.paths["centerlines_resampled"])
        existing = set(os.listdir(self.paths["centerlines_resampled_obj"]))
        for file in centerlines:
            if not file.endswith(".vtp"):
                continue
            out_name = os.path.splitext(file)[0] + ".obj"
            if out_name in existing:
                continue
            print("Converting centerline to obj:", file)
            vtpToObj(file, self.paths["centerlines_resampled"], self.paths["centerlines_resampled_obj"])

    def vessels_to_obj(self):
        self._ensure_dir(self.paths["vessels_obj"])
        vessels = os.listdir(self.paths["vessels_normalized"])
        existing = set(os.listdir(self.paths["vessels_obj"]))
        for file in vessels:
            if not file.endswith(".vtp"):
                continue
            out_name = os.path.splitext(file)[0] + ".obj"
            if out_name in existing:
                continue
            print("Converting vessel to obj:", file)
            vtpToObj(file, self.paths["vessels_normalized"], self.paths["vessels_obj"])

    def build_radius_arrays(self):
        self._ensure_dir(self.paths["radius_arrays"])
        reader = vtk.vtkXMLPolyDataReader()

        for filename in os.listdir(self.paths["centerlines_resampled"]):
            if not filename.endswith(".vtp"):
                continue
            filepath = os.path.join(self.paths["centerlines_resampled"], filename)
            reader.SetFileName(filepath)
            reader.Update()
            polydata = reader.GetOutput()
            if _is_empty_polydata(polydata):
                print("Skipping radius extraction for empty centerline:", filename)
                continue

            point_data = polydata.GetPointData()
            radius_array = point_data.GetArray("Radius")
            if radius_array is None:
                print("No 'Radius' array found in", filename)
                continue

            radius_np = vtk_to_numpy(radius_array)
            output_filename = os.path.splitext(filename)[0] + "_radius.npy"
            output_path = os.path.join(self.paths["radius_arrays"], output_filename)
            if os.path.exists(output_path):
                continue
            np.save(output_path, radius_np)
            print("Saved:", output_path)

    def build_splines(self):
        self._ensure_dir(self.paths["splines_coef"])
        self._ensure_dir(self.paths["splines_knots"])

        centerfolder = self._ensure_trailing_sep(self.paths["centerlines_resampled"])
        meshfolder = self._ensure_trailing_sep(self.paths["vessels_normalized"])
        coef_folder = self._ensure_trailing_sep(self.paths["splines_coef"])

        meshes = sorted(os.listdir(meshfolder))
        for mesh in meshes:
            if not mesh.endswith(".vtp"):
                continue
            base = os.path.splitext(mesh)[0]
            centerline_path = os.path.join(self.paths["centerlines_resampled"], base + "-network.vtp")
            if not os.path.exists(centerline_path):
                print("Skipping splines (missing centerline):", mesh)
                continue
            try:
                center_poly = _read_vtp_polydata(centerline_path)
            except Exception:
                center_poly = None
            if _is_empty_polydata(center_poly):
                print("Skipping splines (empty centerline):", mesh)
                continue
            calculate_splines(mesh, coef_folder, centerfolder, meshfolder, params=self.params)

    def build_graphs(self):
        self._ensure_dir(self.paths["grafos"])
        gfolder = set(os.listdir(self.paths["grafos"]))
        mesh_obj = os.listdir(self.paths["vessels_obj"])

        for file in mesh_obj:
            if not file.endswith(".obj"):
                continue
            graph_name = os.path.splitext(file)[0] + "-grafo.gpickle"
            if graph_name in gfolder:
                continue
            try:
                center_obj = os.path.join(
                    self.paths["centerlines_resampled_obj"], os.path.splitext(file)[0] + "-network.obj"
                )
                radius_path = os.path.join(
                    self.paths["radius_arrays"], os.path.splitext(file)[0] + "-network_radius.npy"
                )
                if not os.path.exists(center_obj) or not os.path.exists(radius_path):
                    print("Skipping graph (missing centerline/radius):", file)
                    continue
                with open(center_obj, "r", encoding="utf-8") as file_obj:
                    grafo = calcularMatriz(file_obj, radius_path)
                print("Calculating graph:", file)
                with open(os.path.join(self.paths["grafos"], graph_name), "wb") as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                print("Problem with:", file)
                traceback.print_exc()

    def build_trees(self):
        self._ensure_dir(self.paths["trees_numpy"])
        self._ensure_dir(self.paths["trees_serialized"])

        t_list = set(os.listdir(self.paths["trees_numpy"]))
        gfolder = set(os.listdir(self.paths["grafos"]))
        files = os.listdir(self.paths["vessels_obj"])

        for file in files:
            if not file.endswith(".obj"):
                continue
            base = os.path.splitext(file)[0]
            graph_name = base + "-grafo.gpickle"
            if graph_name not in gfolder:
                continue
            if base + ".npy" in t_list:
                continue

            try:
                grafo = pickle.load(open(os.path.join(self.paths["grafos"], graph_name), "rb"))
                grafo = grafo.to_undirected()

                if len(nx.cycle_basis(grafo)) > 0:
                    print("Graph has cycles:", file)
                    continue

                for nodo in grafo.nodes:
                    if len(grafo.edges(nodo)) > 3:
                        binarizar(grafo)
                        break

                a_recorrer = []
                numero_nodo_inicial = 1
                distancias = nx.floyd_warshall(grafo)

                par_maximo = (-1, -1)
                maxima = -1
                for nodo_inicial in distancias.keys():
                    for nodo_final in distancias[nodo_inicial]:
                        if distancias[nodo_inicial][nodo_final] > maxima:
                            maxima = distancias[nodo_inicial][nodo_final]
                            par_maximo = (nodo_inicial, nodo_final)

                for nodo in grafo.nodes:
                    if distancias[par_maximo[0]][nodo] == int(maxima / 2):
                        numero_nodo_inicial = nodo
                        if len(grafo.edges(numero_nodo_inicial)) > 2:
                            numero_nodo_inicial = list(grafo.edges(numero_nodo_inicial))[0][1]
                        break

                rad = list(grafo.nodes[numero_nodo_inicial]["radio"])
                nodo_raiz = modelo.Node(numero_nodo_inicial, radius=rad)

                for vecino in grafo.neighbors(numero_nodo_inicial):
                    if vecino != numero_nodo_inicial:
                        a_recorrer.append((vecino, numero_nodo_inicial, nodo_raiz))

                while a_recorrer:
                    nodo_agregar, nodo_padre_id, nodo_padre = a_recorrer.pop(0)
                    radius = list(grafo.nodes[nodo_agregar]["radio"])
                    nodo_actual = modelo.Node(nodo_agregar, radius=radius)
                    nodo_padre.agregarHijo(nodo_actual)
                    for vecino in grafo.neighbors(nodo_agregar):
                        if vecino != nodo_padre_id:
                            a_recorrer.append((vecino, nodo_agregar, nodo_actual))

                serial = grafo2arbol(grafo)
                f = []
                traversefeaturesSerializado(nodo_raiz, f, k=4)
                array = np.array(f)
                np.save(os.path.join(self.paths["trees_numpy"], base), array)
                print("Calculated tree:", file)

                tree_path = os.path.join(self.paths["trees_serialized"], base + "_tree.dat")
                with open(tree_path, "w", encoding="utf-8") as out_f:
                    out_f.write(serial)
            except Exception:
                print("Error with:", file)
                traceback.print_exc()

    def build_graphs_splines(self):
        self._ensure_dir(self.paths["grafos_splines"])
        gfolder = set(os.listdir(self.paths["grafos_splines"]))
        files = os.listdir(self.paths["vessels_obj"])

        for file in files:
            if not file.endswith(".obj"):
                continue
            graph_name = os.path.splitext(file)[0] + "-grafo.gpickle"
            if graph_name in gfolder:
                continue
            try:
                obj_path = os.path.join(
                    self.paths["centerlines_resampled_obj"], os.path.splitext(file)[0] + "-network.obj"
                )
                coef_path = os.path.join(self.paths["splines_coef"], os.path.splitext(file)[0] + ".pkl")
                if not os.path.exists(obj_path) or not os.path.exists(coef_path):
                    print("Skipping spline graph (missing centerline/spline):", file)
                    continue
                with open(obj_path, "r", encoding="utf-8") as file_obj:
                    grafo = calcularMatrizSplines(file_obj, coef_path)
                print("Calculating spline graph:", file)
                with open(os.path.join(self.paths["grafos_splines"], graph_name), "wb") as f:
                    pickle.dump(grafo, f, pickle.HIGHEST_PROTOCOL)
            except Exception:
                print("Problem with:", file)
                traceback.print_exc()

    def build_trees_splines(self):
        self._ensure_dir(self.paths["trees_splines"])
        t_list = set(os.listdir(self.paths["trees_splines"]))
        gfolder = set(os.listdir(self.paths["grafos_splines"]))
        files = os.listdir(self.paths["vessels_obj"])
        n_cp = int(self.params.get("n_cp", 8))
        n_knot = n_cp + 4
        k = 3 + 3 * n_cp + n_knot

        for file in files:
            if not file.endswith(".obj"):
                continue
            base = os.path.splitext(file)[0]
            graph_name = base + "-grafo.gpickle"
            if graph_name not in gfolder:
                continue
            if base + ".npy" in t_list:
                continue

            try:
                grafo = pickle.load(open(os.path.join(self.paths["grafos_splines"], graph_name), "rb"))
                grafo = grafo.to_undirected()

                if len(nx.cycle_basis(grafo)) > 0:
                    print("Graph has cycles:", file)
                    continue

                for nodo in grafo.nodes:
                    if len(grafo.edges(nodo)) > 3:
                        binarizar(grafo)
                        break

                a_recorrer = []
                numero_nodo_inicial = 1
                distancias = nx.floyd_warshall(grafo)

                par_maximo = (-1, -1)
                maxima = -1
                for nodo_inicial in distancias.keys():
                    for nodo_final in distancias[nodo_inicial]:
                        if distancias[nodo_inicial][nodo_final] > maxima:
                            maxima = distancias[nodo_inicial][nodo_final]
                            par_maximo = (nodo_inicial, nodo_final)

                for nodo in grafo.nodes:
                    if distancias[par_maximo[0]][nodo] == int(maxima / 2):
                        numero_nodo_inicial = nodo
                        if len(grafo.edges(numero_nodo_inicial)) > 2:
                            numero_nodo_inicial = list(grafo.edges(numero_nodo_inicial))[0][1]
                        break

                rad = list(grafo.nodes[numero_nodo_inicial]["radio"])
                rad = limpiarRadiosSplines(rad, n_cp=n_cp)
                nodo_raiz = modelo.Node(numero_nodo_inicial, radius=rad)

                for vecino in grafo.neighbors(numero_nodo_inicial):
                    if vecino != numero_nodo_inicial:
                        a_recorrer.append((vecino, numero_nodo_inicial, nodo_raiz))

                while a_recorrer:
                    nodo_agregar, nodo_padre_id, nodo_padre = a_recorrer.pop(0)
                    radius = list(grafo.nodes[nodo_agregar]["radio"])
                    radius = limpiarRadiosSplines(radius, n_cp=n_cp)
                    nodo_actual = modelo.Node(nodo_agregar, radius=radius)
                    nodo_padre.agregarHijo(nodo_actual)
                    for vecino in grafo.neighbors(nodo_agregar):
                        if vecino != nodo_padre_id:
                            a_recorrer.append((vecino, nodo_agregar, nodo_actual))

                f = []
                traversefeaturesSerializado(nodo_raiz, f, k=k)
                array = np.array(f)
                np.save(os.path.join(self.paths["trees_splines"], base), array)
                print("Calculated spline tree:", file)
            except Exception:
                print("Error with:", file)
                traceback.print_exc()

    def copy_raw_data(self):
        self._ensure_dir(self.paths["vessels_normalized"])
        overwrite = bool(self.params.get("normalize_overwrite", False))
        writer = vtk.vtkXMLPolyDataWriter()
        meshes = self._discover_raw_meshes()
        print("Found raw meshes for copy_raw_data:", len(meshes))
        copied = 0

        for case_id, src_path in meshes:
            dst_path = join(self.paths["vessels_normalized"], case_id + ".vtp")
            if os.path.exists(dst_path) and not overwrite:
                continue
            ext = os.path.splitext(src_path)[1].lower()
            if ext == ".vtp":
                shutil.copy2(src_path, dst_path)
                copied += 1
                continue
            if ext == ".obj":
                try:
                    polydata = self._read_polydata(src_path)
                except Exception:
                    print("Could not read mesh:", src_path)
                    traceback.print_exc()
                    continue
                # Merge duplicate vertices (raw OBJ exports often have one
                # vertex per triangle corner, which breaks downstream vmtk).
                try:
                    cleaner = vtk.vtkCleanPolyData()
                    cleaner.SetInputData(polydata)
                    cleaner.SetTolerance(0.0)
                    cleaner.PointMergingOn()
                    cleaner.ConvertLinesToPointsOff()
                    cleaner.ConvertPolysToLinesOff()
                    cleaner.ConvertStripsToPolysOff()
                    cleaner.Update()
                    cleaned = vtk.vtkPolyData()
                    cleaned.DeepCopy(cleaner.GetOutput())
                    if cleaned.GetNumberOfPoints() > 0 and cleaned.GetNumberOfCells() > 0:
                        polydata = cleaned
                except Exception:
                    print("vtkCleanPolyData failed in copy_raw_data:", src_path)
                    traceback.print_exc()
                writer.SetFileName(dst_path)
                writer.SetInputData(polydata)
                writer.Write()
                copied += 1

        print("copy_raw_data: prepared", copied, "raw meshes in vessels_normalized")
        return

    def run(self, only_steps=None):
        self._validate_paths()

        step_funcs = {
            "copy_raw_data": self.copy_raw_data,  # Placeholder if needed
            "normalize_meshes": self.normalize_meshes,
            "extract_centerlines": self.extract_centerlines,
            "resample_centerlines": self.resample_centerlines,
            "centerlines_to_obj": self.centerlines_to_obj,
            "vessels_to_obj": self.vessels_to_obj,
            "radius_arrays": self.build_radius_arrays,
            "splines": self.build_splines,
            "graphs": self.build_graphs,
            "trees": self.build_trees,
            "graphs_splines": self.build_graphs_splines,
            "trees_splines": self.build_trees_splines,
        }

        if only_steps:
            steps_to_run = [s for s in self.STEP_ORDER if s in set(only_steps)]
        else:
            steps_to_run = [s for s in self.STEP_ORDER if self.flags.get(s, False)]

        for step in steps_to_run:
            print("=== Running step:", step)
            step_funcs[step]()


def load_config(path):
    return PreprocessingPipeline.load_config(path)


def run_pipeline(cfg, only_steps=None):
    pipeline = PreprocessingPipeline(cfg)
    pipeline.run(only_steps=only_steps)
