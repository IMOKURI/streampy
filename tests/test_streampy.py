import numpy as np

from streampy import StreamPy


def test_construct():
    s = StreamPy.empty(["a", "b"], np.float32)

    assert s.values.shape == (1000, 2)
    assert s.dtype == np.float32
    assert s.values[0][0] == 0.0
    assert s.length == 0
    assert s.default_value is None
    assert s.capacity == 1000
    assert s.columns == ["a", "b"]


def test_extend_basic():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    assert s.length == 2
    assert s.values[0][0] == 1.0
    assert s.values[1][1] == 4.0


def test_extend_nan():
    s = StreamPy.empty(["a", "b"], np.float32, default_value=-1.0)
    array = np.array([[1, np.nan]], dtype=np.float32)
    s.extend(array)

    assert s.length == 1
    assert s.values[0][0] == 1.0
    assert s.values[0][1] == -1.0


def test_extend_object():
    s = StreamPy.empty(["a", "b", "c"], object, default_value=None)
    array = np.array([["hello", True, np.nan]], dtype=object)
    s.extend(array)

    assert s.length == 1
    assert s.values[0][0] == "hello"
    assert s.values[0][1]
    assert s.values[0][2] is None


def test_extend_grow():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.ones((999, 2), dtype=np.float32)
    s.extend(array)

    assert s.length == 999

    array = np.array([[1, 2]], dtype=np.float32)
    s.extend(array)

    assert s.length == 1000
    assert s.capacity == 1500


def test_extend_grow_large():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.ones((999, 2), dtype=np.float32)
    s.extend(array)

    assert s.length == 999

    array = np.ones((1000, 2), dtype=np.float32)
    s.extend(array)

    assert s.length == 1999
    assert s.capacity == 1999


def test_last_n_basic():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    res = s.last_n(1)

    assert res.shape == (1, 2)
    assert res[0][1] == 4.0


def test_last_n_large():
    s = StreamPy.empty(["a", "b"], np.float32)
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    s.extend(array)

    res = s.last_n(3)

    assert res.shape == (3, 2)
    assert res[0][0] == 0.0
    assert res[1][0] == 1.0


def test_recent_n_days_index():
    s = StreamPy.empty(["Date"], "datetime64[D]", default_value=np.datetime64("NaT"))
    array = np.array(
        [
            ["2022-03-30"],
            ["2022-03-31"],
            ["2022-04-01"],
            ["2022-04-02"],
            ["2022-04-03"],
            ["2022-04-04"],
            ["2022-04-05"],
        ],
        dtype="datetime64[D]",
    )
    s.extend(array)

    res = s.recent_n_days_index(3, np.datetime64("2022-04-03", "D"))

    assert np.array_equal(res, [1, 2, 3])


def test_recent_n_days_index_include_base():
    s = StreamPy.empty(["Date"], "datetime64[D]", default_value=np.datetime64("NaT"))
    array = np.array(
        [
            ["2022-03-30"],
            ["2022-03-31"],
            ["2022-04-01"],
            ["2022-04-02"],
            ["2022-04-03"],
            ["2022-04-04"],
            ["2022-04-05"],
        ],
        dtype="datetime64[D]",
    )
    s.extend(array)

    res = s.recent_n_days_index(3, np.datetime64("2022-04-03", "D"), include_base=True)

    assert np.array_equal(res, [1, 2, 3, 4])


def test_last_n_days_index():
    s = StreamPy.empty(["Date"], "datetime64[D]", default_value=np.datetime64("NaT"))
    array = np.array(
        [
            ["2022-03-31"],
            ["2022-04-01"],
            ["2022-04-02"],
            ["2022-04-03"],
            ["2022-04-04"],
            ["2022-04-05"],
        ],
        dtype="datetime64[D]",
    )
    s.extend(array)

    res = s.last_n_days_index(3)

    assert np.array_equal(res, [2, 3, 4])


def test_last_n_days_index_include_base():
    s = StreamPy.empty(["Date"], "datetime64[D]", default_value=np.datetime64("NaT"))
    array = np.array(
        [
            ["2022-03-31"],
            ["2022-04-01"],
            ["2022-04-02"],
            ["2022-04-03"],
            ["2022-04-04"],
            ["2022-04-05"],
        ],
        dtype="datetime64[D]",
    )
    s.extend(array)

    res = s.last_n_days_index(3, include_base=True)

    assert np.array_equal(res, [2, 3, 4, 5])
