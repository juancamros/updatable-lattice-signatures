import copy
import numpy as np
import pytest
from src.crsst21 import crsst21, generar_palabras

SEED = 2026
N = 5
C_UPD = 22900


@pytest.fixture
def system_setup():
    s = crsst21(32, N, C_UPD, seed=SEED)
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

    s_esperados = [np.round(s.p0_param * s.w_param, 2)]
    for _ in range(s.N_param - 1):
        prev = s_esperados[-1]
        s_esperados.append(np.round(s.cupd_param * prev + s.cupd_param, 2))
    np.testing.assert_allclose(s.s_params, s_esperados)

    beta0 = np.round(1.5 * s.r_param * s.s_params[0] * np.sqrt(s.m_param), 2)
    beta_esperados = [beta0]
    for _ in range(s.N_param - 1):
        beta_esperados.append(np.round(s.cupd_param * beta_esperados[-1], 2))
    np.testing.assert_allclose(s.beta_params, beta_esperados)


def test_sigma_epoch_inicial_corresponde_a_s0(keyed_system):
    s, _, _ = keyed_system
    esperada = s.s_params[0] ** 2 * np.eye(s.m_param)
    np.testing.assert_allclose(s.SIGMA, esperada)


def test_gram_schmidt_ortogonaliza():
    base = np.array([[1.0, 1.0], [0.0, 1.0]])
    b = crsst21.gram_schmidt(base)
    assert np.dot(b[:, 0], b[:, 1]) == pytest.approx(0.0, abs=1e-12)


def test_gram_schmidt_detecta_dependencia():
    base = np.array([[1.0, 2.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="linealmente dependientes"):
        crsst21.gram_schmidt(base)


def test_trapgen_dimensiones_y_R(keyed_system):
    s, A, R = keyed_system
    assert A.shape == (s.n_param, s.m_param)
    assert R.shape == (s.mbar_param, s.w_param)
    assert set(np.unique(R)).issubset({-1, 0, 1})


def test_trapdoor_AB_igual_G(keyed_system):
    s, A, _ = keyed_system
    np.testing.assert_array_equal((A @ s.B_matrix) % s.q_param, s.G_matrix % s.q_param)


def test_S_en_kernel_de_G(keyed_system):
    s, _, _ = keyed_system
    assert np.all((s.G_matrix @ s.S_matrix) % s.q_param == 0)


@pytest.mark.parametrize("columnas", [1, 3])
def test_bitdecomp_reconstruye_V(keyed_system, columnas):
    s, _, _ = keyed_system
    rng = np.random.default_rng(1234)
    V = rng.integers(0, s.q_param, size=(s.n_param, columnas), dtype=np.int64)
    Z0 = s.BitDecomp(V)
    np.testing.assert_array_equal((s.G_matrix @ Z0) % s.q_param, V % s.q_param)


def test_oracle_sampler_resuelve_GZ_igual_V(keyed_system):
    s, _, _ = keyed_system
    V = np.column_stack([s.H("a"), s.H("b")])
    Z = s.oracle_sampler(V)
    np.testing.assert_array_equal((s.G_matrix @ Z) % s.q_param, V % s.q_param)


def test_samplepre_sign_resuelve_AX_igual_Y(keyed_system):
    s, A, _ = keyed_system
    Y = np.column_stack([s.H("mensaje-1"), s.H("mensaje-2")])
    X = s.SamplePre(Y, mode="sign")
    np.testing.assert_array_equal((A @ X) % s.q_param, Y % s.q_param)


def test_samplepre_sign_usa_s0_incluso_en_epoca_1(keyed_system):
    s, pk0, _ = keyed_system
    pk1, _, _ = s.Next(pk0)
    Y = s.H("firma fresca en epoca 1")
    s.SamplePre(Y, mode="sign")

    B = s.B_matrix.astype(float)
    mat_g = B @ s.SIGMA_G @ B.T
    esperada = s.s_params[0] ** 2 * s.I_m.astype(float) - mat_g
    esperada = (esperada + esperada.T) / 2
    np.testing.assert_allclose(s.SIGMA_p, esperada, rtol=1e-12, atol=1e-10)


def test_samplepre_token_usa_covarianza_especial_y_es_psd(keyed_system):
    s, A, _ = keyed_system
    Y = A[:, :3]
    X = s.SamplePre(Y, mode="token")
    np.testing.assert_array_equal((A @ X) % s.q_param, Y % s.q_param)

    B = s.B_matrix.astype(float)
    mat_g = B @ s.SIGMA_G @ B.T
    delta = 0.02
    eta = s.sigma_G
    boundary = B @ (2 * s.I_w + s.SIGMA_G) @ B.T
    sigma_token = (1 + delta) * boundary + eta * s.I_m
    esperada = sigma_token - mat_g
    esperada = (esperada + esperada.T) / 2
    np.testing.assert_allclose(s.SIGMA_p, esperada, rtol=1e-12, atol=1e-10)
    assert np.linalg.eigvalsh(s.SIGMA_p).min() >= -1e-8


def test_samplepre_rechaza_modo_invalido(keyed_system):
    s, _, _ = keyed_system
    with pytest.raises(ValueError, match="modo válido"):
        s.SamplePre(s.H("x"), mode="otro")


def test_trapgen_debe_validar_s0_para_firmas_frescas(system_setup):
    """CRSST21 firma fresco con s0; TrapGen no debe validar sólo s_e si e>0."""
    s = system_setup
    s.epoc_param = 1
    s.s_params[0] = 0.01
    s.s_params[1] = 1e12
    with pytest.raises(ValueError, match="s_required_2"):
        s.TrapGen()


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

    def fake_samplepre(Y, mode="sign"):
        usado["mode"] = mode
        return token

    monkeypatch.setattr(s, "KeyGen", fake_keygen)
    monkeypatch.setattr(s, "SamplePre", fake_samplepre)
    s.Next(pk_old)
    assert usado["mode"] == "token"


@pytest.mark.integration
def test_firma_y_verificacion(keyed_system):
    s, pk, _ = keyed_system
    sign = s.Sig("mensaje correcto")
    assert s.Ver(pk, "mensaje correcto", sign)
    assert not s.Ver(pk, "mensaje alterado", sign)


@pytest.mark.integration
def test_next_genera_token_y_actualiza_sigma(keyed_system):
    s, pk0, _ = keyed_system
    pk1, _, token = s.Next(pk0)
    assert s.epoc_param == 1
    assert token.shape == (s.m_param, s.m_param)
    np.testing.assert_array_equal((pk1 @ token) % s.q_param, pk0 % s.q_param)
    np.testing.assert_allclose(s.SIGMA, s.s_params[1] ** 2 * np.eye(s.m_param))
    assert np.linalg.norm(token.astype(float), 2) <= s.cupd_param


@pytest.mark.integration
def test_update_es_exactamente_determinista_y_no_mueve_rng(keyed_system):
    s, pk0, _ = keyed_system
    sign0 = s.Sig("determinismo")
    pk1, _, token = s.Next(pk0)

    estado_antes = copy.deepcopy(s.rng.bit_generator.state)
    esperado_big = np.asarray(token, dtype=object) @ np.asarray(sign0, dtype=object)
    assert max(abs(int(x)) for x in esperado_big) <= np.iinfo(np.int64).max

    sign1 = s.Update(token, sign0)
    estado_despues = s.rng.bit_generator.state

    np.testing.assert_array_equal(np.asarray(sign1, dtype=object), esperado_big)
    assert estado_antes == estado_despues
    assert s.Ver(pk1, "determinismo", sign1)


@pytest.mark.integration
def test_firma_fresca_epoca_1_sigue_usando_s0(keyed_system):
    s, pk0, _ = keyed_system
    s.Next(pk0)
    sign = s.Sig("fresca-e1")
    assert s.Ver(s.pk, "fresca-e1", sign)

    B = s.B_matrix.astype(float)
    mat_g = B @ s.SIGMA_G @ B.T
    esperada = s.s_params[0] ** 2 * s.I_m.astype(float) - mat_g
    esperada = (esperada + esperada.T) / 2
    np.testing.assert_allclose(s.SIGMA_p, esperada, rtol=1e-12, atol=1e-10)


@pytest.mark.integration
def test_cadena_determinista_todas_las_epocas_sin_overflow(keyed_system):
    s, pk, _ = keyed_system
    mensaje = "cadena completa"
    sign = s.Sig(mensaje)
    assert s.Ver(pk, mensaje, sign)

    for _ in range(1, s.N_param):
        old_pk = pk
        old_sign = sign
        pk, _, token = s.Next(old_pk)
        assert np.linalg.norm(token.astype(float), 2) <= s.cupd_param

        esperado_big = np.asarray(token, dtype=object) @ np.asarray(old_sign, dtype=object)
        assert max(abs(int(x)) for x in esperado_big) <= np.iinfo(np.int64).max

        sign = s.Update(token, old_sign)
        np.testing.assert_array_equal(np.asarray(sign, dtype=object), esperado_big)
        assert s.Ver(pk, mensaje, sign)


def test_update_en_epoca_inicial_falla(system_setup):
    with pytest.raises(ValueError, match="época inicial"):
        system_setup.Update(np.empty((0, 0), dtype=np.int64), np.empty(0, dtype=np.int64))


def test_next_rechaza_pk_incorrecta(keyed_system):
    s, pk, _ = keyed_system
    mala = pk.copy()
    mala[0, 0] = (mala[0, 0] + 1) % s.q_param
    with pytest.raises(ValueError, match="no coincide"):
        s.Next(mala)


def test_generar_palabras_reproducible():
    a = generar_palabras(10, 7)
    b = generar_palabras(10, 7)
    assert a == b
    assert len(a) == 10
