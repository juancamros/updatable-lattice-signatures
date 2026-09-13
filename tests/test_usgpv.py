import numpy as np
import pytest

from src.usgpv import usgpv, generar_palabras


SEED = 2026
N = 4  # evitamos la época 4 con los parámetros actuales: excede el rango fiable de int64
C_UPD = 22900


@pytest.fixture
def system_setup():
    s = usgpv(32, N, C_UPD, seed=SEED)
    s.setup()
    return s


@pytest.fixture
def keyed_system(system_setup):
    pk, sk = system_setup.KeyGen()
    return system_setup, pk, sk


def test_setup_parametros_por_epoca(system_setup):
    s = system_setup
    assert len(s.s_params) == s.N_param
    assert len(s.beta_params) == s.N_param
    assert all(x > 0 for x in s.s_params)
    assert all(x > 0 for x in s.beta_params)


def test_trapdoor_AB_igual_G(keyed_system):
    s, A, _ = keyed_system
    np.testing.assert_array_equal((A @ s.B_matrix) % s.q_param, s.G_matrix % s.q_param)


def test_S_en_kernel_de_G(keyed_system):
    s, _, _ = keyed_system
    assert np.all((s.G_matrix @ s.S_matrix) % s.q_param == 0)


def test_samplepre_sign_usa_dispersion_de_epoca(keyed_system):
    s, pk0, _ = keyed_system
    s.VE_impostor_KeyGen()
    s.Next(pk0)
    s.SamplePre(s.H("e1"), mode="sign")

    B = s.B_matrix.astype(float)
    mat_g = B @ s.SIGMA_G @ B.T
    esperada = s.s_params[1] ** 2 * s.I_m.astype(float) - mat_g
    esperada = (esperada + esperada.T) / 2
    np.testing.assert_allclose(s.SIGMA_p, esperada, rtol=1e-12, atol=1e-8)


def test_samplepre_token_covarianza_especial(keyed_system):
    s, A, _ = keyed_system
    s.SamplePre(A[:, :2], mode="token")
    B = s.B_matrix.astype(float)
    mat_g = B @ s.SIGMA_G @ B.T
    boundary = B @ (2 * s.I_w + s.SIGMA_G) @ B.T
    sigma_token = 1.02 * boundary + s.sigma_G * s.I_m
    esperada = sigma_token - mat_g
    esperada = (esperada + esperada.T) / 2
    np.testing.assert_allclose(s.SIGMA_p, esperada, rtol=1e-12, atol=1e-10)


def test_next_usa_modo_token(monkeypatch, system_setup):
    s = system_setup
    pk_old = np.zeros((s.n_param, s.m_param), dtype=np.int64)
    s.pk = pk_old.copy()
    pk_new = np.zeros_like(pk_old)
    sk_new = np.zeros((s.mbar_param, s.w_param), dtype=np.int64)
    token = np.zeros((s.m_param, s.m_param), dtype=np.int64)
    usado = {}

    def fake_keygen():
        s.pk, s.sk = pk_new, sk_new
        return s.pk, s.sk

    def fake_ve_keygen():
        s.VE_pk = object()
        return s.VE_pk

    def fake_samplepre(Y, mode="sign"):
        usado["mode"] = mode
        return token

    monkeypatch.setattr(s, "KeyGen", fake_keygen)
    monkeypatch.setattr(s, "VE_impostor_KeyGen", fake_ve_keygen)
    monkeypatch.setattr(s, "SamplePre", fake_samplepre)
    s.Next(pk_old)
    assert usado["mode"] == "token"


@pytest.mark.integration
def test_flujo_firma_update_verificacion(keyed_system):
    s, pk0, _ = keyed_system
    vepk0 = s.VE_impostor_KeyGen()
    msg = "mensaje usgpv"
    sign0 = s.Sig(msg)
    assert s.Ver(vepk0, pk0, msg, sign0)

    pk1, _, token = s.Next(pk0)
    sign1 = s.Update(token, sign0)
    assert s.Ver(s.VE_pk, pk1, msg, sign1)

    np.testing.assert_array_equal(
        ((pk1 @ sign1[0]) - sign1[1]) % s.q_param,
        ((pk0 @ sign0[0]) - sign0[1]) % s.q_param,
    )


def test_parametros_actuales_epoca_4_exceden_escala_int64():
    """Documenta una limitación numérica actual: no es un fallo criptográfico, sino de dtype."""
    s = usgpv(32, 5, C_UPD, seed=SEED)
    s.setup()
    escala = s.r_param * s.s_params[4]
    assert escala > np.iinfo(np.int64).max
