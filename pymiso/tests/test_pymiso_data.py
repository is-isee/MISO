from pathlib import Path
import struct

import numpy as np

import pymiso
from pymiso import Conf, Data, Grid, MPI, Time


def _write_config(base_dir: Path, body: str) -> None:
    (base_dir / "config.yaml").write_text(body, encoding="utf-8")


def _write_grid(base_dir: Path, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> None:
    with (base_dir / "grid.bin").open("wb") as f:
        f.write(struct.pack("<I", 4))
        x.astype("<f4").tofile(f)
        y.astype("<f4").tofile(f)
        z.astype("<f4").tofile(f)


def _write_coords(base_dir: Path, rows: list[tuple[int, int, int, int]]) -> None:
    mpi_dir = base_dir / "mpi"
    mpi_dir.mkdir()
    with (mpi_dir / "coords.csv").open("w", encoding="utf-8") as f:
        f.write("rank,x,y,z\n")
        for row in rows:
            f.write("{},{},{},{}\n".format(*row))


def _write_time(base_dir: Path, n_output_digits: int, time_value: float = 0.0) -> None:
    time_dir = base_dir / "time"
    time_dir.mkdir()
    (time_dir / "n_output.txt").write_text("0\n", encoding="utf-8")
    filename = time_dir / f"time.{0:0{n_output_digits}d}.txt"
    filename.write_text(f"{time_value}\n0\n0\n", encoding="utf-8")


def _write_mhd_rank(
    filename: Path,
    arrays: list[np.ndarray],
) -> None:
    with filename.open("wb") as f:
        f.write(struct.pack("<I", 4))
        for arr in arrays:
            arr.astype("<f4").ravel(order="C").tofile(f)


def _write_rt_rank(
    filename: Path,
    num_rays: int,
    weights: np.ndarray,
    mu_x: np.ndarray,
    mu_y: np.ndarray,
    mu_z: np.ndarray,
    src_func: np.ndarray,
    abs_coeff: np.ndarray,
    rint: np.ndarray,
) -> None:
    with filename.open("wb") as f:
        f.write(struct.pack("<i", num_rays))
        weights.astype("<f4").tofile(f)
        mu_x.astype("<f4").tofile(f)
        mu_y.astype("<f4").tofile(f)
        mu_z.astype("<f4").tofile(f)
        src_func.astype("<f4").ravel(order="C").tofile(f)
        abs_coeff.astype("<f4").ravel(order="C").tofile(f)
        rint.astype("<f4").ravel(order="C").tofile(f)


def test_package_exports_public_api() -> None:
    assert pymiso.__all__ == ["Conf", "Data", "Grid", "MPI", "Time"]
    assert pymiso.Conf is Conf
    assert pymiso.Data is Data
    assert pymiso.Grid is Grid
    assert pymiso.MPI is MPI
    assert pymiso.Time is Time


def test_conf_sets_defaults_and_paths(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )

    conf = Conf(str(tmp_path))

    assert conf.physics.mhd is True
    assert conf.physics.rt is False
    assert conf.time_data_dir == tmp_path / "time"
    assert conf.mhd_data_dir == tmp_path / "mhd"
    assert conf.mpi_data_dir == tmp_path / "mpi"
    assert conf.endian == "<"


def test_conf_rt_physics_override_is_preserved(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
physics:
  rt: true
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 4
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 4.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 2
  y_procs: 1
  z_procs: 1
rt:
  save_dir: rt/
  num_rays: 2
data_type:
  Endian: little
""".strip()
        + "\n",
    )

    conf = Conf(str(tmp_path))

    assert conf.physics.mhd is False
    assert conf.physics.rt is True
    assert conf.rt_data_dir == tmp_path / "rt"


def test_grid_loads_geometry_and_edges(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 4
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 4.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 2
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )

    conf = Conf(str(tmp_path))
    grid = Grid(conf)

    np.testing.assert_allclose(grid.x, np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32))
    np.testing.assert_allclose(grid.y, np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(grid.z, np.array([0.5], dtype=np.float32))
    np.testing.assert_allclose(
        grid.x_edge, np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    )
    assert grid.i_total == 6
    assert grid.j_total == 4
    assert grid.k_total == 1
    assert grid.i_size_local == 2
    assert grid.i_total_local == 4


def test_mpi_loads_rank_coordinates(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 4
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 4.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 2
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_coords(tmp_path, [(0, 0, 0, 0), (1, 1, 0, 0)])

    conf = Conf(str(tmp_path))
    mpi = MPI(conf)

    assert mpi.n_procs == 2
    np.testing.assert_array_equal(mpi.coords["rank"], np.array([0, 1]))
    np.testing.assert_array_equal(mpi.coords["x"], np.array([0, 1]))
    np.testing.assert_array_equal(mpi.coords["y"], np.array([0, 0]))
    np.testing.assert_array_equal(mpi.coords["z"], np.array([0, 0]))


def test_time_load_reads_snapshot_metadata(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 2.0
  dt_output: 0.5
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_time(tmp_path, n_output_digits=8, time_value=1.5)
    time_dir = tmp_path / "time"
    (time_dir / "n_output.txt").write_text("3\n", encoding="utf-8")
    (time_dir / "time.00000002.txt").write_text("1.5\n2\n7\n", encoding="utf-8")

    conf = Conf(str(tmp_path))
    time = Time(conf)
    time.load(2)

    assert time.n_output == 3
    assert time.time == 1.5
    assert time.load_n_output == 2
    assert time.load_n_step == 7


def test_data_loads_mhd_fixture(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
physics:
  mhd: true
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0)])
    _write_time(tmp_path, n_output_digits=8)
    (tmp_path / "mhd").mkdir()

    shape_local = (4, 4, 1)
    base = np.arange(np.prod(shape_local), dtype=np.float32).reshape(shape_local)
    arrays = [base + 100.0 * i for i in range(9)]
    _write_mhd_rank(tmp_path / "mhd" / "mhd.00000000.00000000.bin", arrays)

    data = Data(str(tmp_path))
    data.load(0)

    expected = base[1:3, 1:3, :]
    assert data.model == "mhd"
    assert data.ro.shape == (2, 2)
    np.testing.assert_allclose(data.ro, expected[:, :, 0])
    np.testing.assert_allclose(data.vx, (expected + 100.0)[:, :, 0])
    assert data.time.time == 0.0
    assert data.load_n_output == 0


def test_data_getattr_delegates_to_grid_time_and_mpi(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
physics:
  mhd: true
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0)])
    _write_time(tmp_path, n_output_digits=8, time_value=0.25)
    (tmp_path / "mhd").mkdir()

    shape_local = (4, 4, 1)
    base = np.arange(np.prod(shape_local), dtype=np.float32).reshape(shape_local)
    arrays = [base + 100.0 * i for i in range(9)]
    _write_mhd_rank(tmp_path / "mhd" / "mhd.00000000.00000000.bin", arrays)

    data = Data(str(tmp_path))
    data.load(0)

    assert data.x_min == 0.0
    assert data.n_procs == 1
    assert data.load_n_step == 0
    np.testing.assert_allclose(data.x, np.array([0.5, 1.5], dtype=np.float32))


def test_data_defaults_to_mhd_when_physics_is_missing(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0)])
    _write_time(tmp_path, n_output_digits=8)
    (tmp_path / "mhd").mkdir()

    shape_local = (4, 4, 1)
    base = np.arange(np.prod(shape_local), dtype=np.float32).reshape(shape_local)
    arrays = [base + 100.0 * i for i in range(9)]
    _write_mhd_rank(tmp_path / "mhd" / "mhd.00000000.00000000.bin", arrays)

    data = Data(str(tmp_path))

    assert data.model == "mhd"
    assert data.conf.physics.mhd is True
    assert data.conf.physics.rt is False


def test_data_detect_model_rejects_invalid_physics(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
physics:
  mhd: false
  rt: false
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0)])
    _write_time(tmp_path, n_output_digits=8)

    try:
        Data(str(tmp_path))
    except ValueError as exc:
        assert "Either physics.mhd or physics.rt must be true" in str(exc)
    else:
        raise AssertionError("Data() should reject invalid physics selection")


def test_data_dtype_from_elem_size_rejects_unknown_size(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 2
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 2.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 1
  y_procs: 1
  z_procs: 1
mhd:
  mhd_save_dir: mhd/
  n_output_digits: 8
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0)])
    _write_time(tmp_path, n_output_digits=8)
    (tmp_path / "mhd").mkdir()

    data = Data(str(tmp_path))

    try:
        data._dtype_from_elem_size(np.uint32(16))
    except ValueError as exc:
        assert "Unexpected element size" in str(exc)
    else:
        raise AssertionError("_dtype_from_elem_size() should reject unknown sizes")


def test_data_loads_rt_fixture(tmp_path: Path) -> None:
    _write_config(
        tmp_path,
        """
base:
  save_dir: data/
physics:
  rt: true
time:
  time_save_dir: time/
  n_output_digits: 8
  tend: 1.0
  dt_output: 1.0
grid:
  i_size: 4
  j_size: 2
  k_size: 1
  margin: 1
  x_min: 0.0
  x_max: 4.0
  y_min: 0.0
  y_max: 2.0
  z_min: 0.0
  z_max: 1.0
mpi:
  mpi_save_dir: mpi/
  x_procs: 2
  y_procs: 1
  z_procs: 1
rt:
  save_dir: rt/
  num_rays: 2
data_type:
  Endian: little
""".strip()
        + "\n",
    )
    _write_grid(
        tmp_path,
        np.array([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float32),
        np.array([-0.5, 0.5, 1.5, 2.5], dtype=np.float32),
        np.array([0.5], dtype=np.float32),
    )
    _write_coords(tmp_path, [(0, 0, 0, 0), (1, 1, 0, 0)])
    _write_time(tmp_path, n_output_digits=8)
    (tmp_path / "rt").mkdir()

    num_rays = 2
    weights = np.array([0.25, 0.75], dtype=np.float32)
    mu_x = np.array([1.0, -1.0], dtype=np.float32)
    mu_y = np.array([0.0, 0.0], dtype=np.float32)
    mu_z = np.array([0.0, 0.0], dtype=np.float32)

    shape_local = (4, 4, 1)
    src_rank0 = np.zeros(shape_local, dtype=np.float32)
    src_rank1 = np.zeros(shape_local, dtype=np.float32)
    abs_rank0 = np.zeros(shape_local, dtype=np.float32)
    abs_rank1 = np.zeros(shape_local, dtype=np.float32)
    src_rank0[1:3, 1:3, 0] = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    src_rank1[1:3, 1:3, 0] = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
    abs_rank0[1:3, 1:3, 0] = 10.0 + src_rank0[1:3, 1:3, 0]
    abs_rank1[1:3, 1:3, 0] = 10.0 + src_rank1[1:3, 1:3, 0]

    rint_rank0 = np.zeros((num_rays, *shape_local), dtype=np.float32)
    rint_rank1 = np.zeros((num_rays, *shape_local), dtype=np.float32)
    rint_rank0[:, 1:3, 1:3, 0] = np.array(
        [[[11.0, 12.0], [13.0, 14.0]], [[21.0, 22.0], [23.0, 24.0]]],
        dtype=np.float32,
    )
    rint_rank1[:, 1:3, 1:3, 0] = np.array(
        [[[15.0, 16.0], [17.0, 18.0]], [[25.0, 26.0], [27.0, 28.0]]],
        dtype=np.float32,
    )

    _write_rt_rank(
        tmp_path / "rt" / "rank_000000.bin",
        num_rays,
        weights,
        mu_x,
        mu_y,
        mu_z,
        src_rank0,
        abs_rank0,
        rint_rank0,
    )
    _write_rt_rank(
        tmp_path / "rt" / "rank_000001.bin",
        num_rays,
        weights,
        mu_x,
        mu_y,
        mu_z,
        src_rank1,
        abs_rank1,
        rint_rank1,
    )

    data = Data(str(tmp_path))
    data.load(0)

    np.testing.assert_allclose(data.weights, weights)
    np.testing.assert_allclose(data.mu_x, mu_x)
    np.testing.assert_allclose(
        data.src_func,
        np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        data.abs_coeff,
        np.array(
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
            dtype=np.float32,
        ),
    )
    assert data.rint.shape == (2, 4, 2)
    np.testing.assert_allclose(
        data.rint[0],
        np.array(
            [[11.0, 12.0], [13.0, 14.0], [15.0, 16.0], [17.0, 18.0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(
        data.rint[1],
        np.array(
            [[21.0, 22.0], [23.0, 24.0], [25.0, 26.0], [27.0, 28.0]],
            dtype=np.float32,
        ),
    )
