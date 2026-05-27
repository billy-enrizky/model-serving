from bench.mfu import GEMMA_4_E2B_N_ACTIVE, compute_mbu, compute_mfu


def test_mfu_zero_throughput():
    r = compute_mfu(tokens_per_sec=0.0, peak_tflops=112.0, n_active_params=GEMMA_4_E2B_N_ACTIVE)
    assert r.achieved_tflops == 0.0
    assert r.mfu == 0.0


def test_mfu_known_value():
    # 100 tok/s on 1.91B params => 2 * 1.91e9 * 100 = 3.82e11 = 0.382 TFLOPS achieved.
    # Peak 112 TFLOPS => MFU = 0.382 / 112 = 0.0034107.
    r = compute_mfu(
        tokens_per_sec=100.0,
        peak_tflops=112.0,
        n_active_params=1_910_000_000,
    )
    assert abs(r.achieved_tflops - 0.382) < 1e-6
    assert abs(r.mfu - (0.382 / 112.0)) < 1e-9
    assert r.flops_per_token == 3_820_000_000


def test_mbu_known_value():
    # TPOT 0.1s, peak 900 GB/s, params 10 GB, no KV.
    # achieved = 10e9 / 0.1 = 1e11 B/s = 100 GB/s. MBU = 100/900.
    r = compute_mbu(
        tpot_seconds=0.1,
        peak_hbm_gbps=900.0,
        param_bytes=10_000_000_000,
        kv_cache_bytes=0,
    )
    assert abs(r.achieved_gbps - 100.0) < 1e-6
    assert abs(r.mbu - (100.0 / 900.0)) < 1e-9


def test_mbu_zero_tpot():
    r = compute_mbu(0.0, 900.0, 10_000_000_000, 0)
    assert r.achieved_gbps == 0.0
    assert r.mbu == 0.0
