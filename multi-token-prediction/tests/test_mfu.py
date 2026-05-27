from bench.mfu import compute_mfu


def test_mfu_zero_throughput():
    r = compute_mfu(tokens_per_sec=0.0, peak_tflops=125.0, n_active_params=2_300_000_000)
    assert r.achieved_tflops == 0.0
    assert r.mfu == 0.0


def test_mfu_known_value():
    # 100 tok/s on 2.3B params => 2 * 2.3e9 * 100 = 4.6e11 = 0.46 TFLOPS achieved.
    # Peak 125 TFLOPS => MFU = 0.46 / 125 = 0.00368.
    r = compute_mfu(
        tokens_per_sec=100.0,
        peak_tflops=125.0,
        n_active_params=2_300_000_000,
    )
    assert abs(r.achieved_tflops - 0.46) < 1e-6
    assert abs(r.mfu - 0.00368) < 1e-5
    assert r.flops_per_token == 4_600_000_000
