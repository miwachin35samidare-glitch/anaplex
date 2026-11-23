"""耳 — メルスペクトログラム。

網膜と蝸牛は遺伝で決まっていて経験で変わらない。可塑なのは皮質から。
だからここは**凍結**で、学習しない（設計資料 §4.1）。

wav2vec 等の学習済みモデルを持ってくる必要がない。固定の信号処理で足りる。
"""

import numpy as np

import config as C


def _hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank(sr, n_fft, n_mels, fmin, fmax):
    n_bins = n_fft // 2 + 1
    mels = np.linspace(_hz_to_mel(fmin), _hz_to_mel(fmax), n_mels + 2)
    hz = _mel_to_hz(mels)
    bin_f = hz * n_fft / sr
    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    k = np.arange(n_bins, dtype=np.float64)
    for i in range(n_mels):
        lo, ctr, hi = bin_f[i], bin_f[i + 1], bin_f[i + 2]
        if ctr > lo:
            rise = (k - lo) / (ctr - lo)
            fb[i] += np.clip(rise, 0.0, 1.0) * ((k >= lo) & (k <= ctr))
        if hi > ctr:
            fall = (hi - k) / (hi - ctr)
            fb[i] += np.clip(fall, 0.0, 1.0) * ((k > ctr) & (k <= hi))
    # Slaney 正規化。帯域幅で割ることで低域が持ち上がりすぎるのを抑える
    widths = hz[2:] - hz[:-2]
    fb *= (2.0 / widths)[:, None]
    return fb


class Ear:
    """凍結された蝸牛。パラメータは一つも学習しない。"""

    def __init__(self):
        self.sr = C.SAMPLE_RATE
        self.n_fft = C.N_FFT
        self.hop = C.HOP_LENGTH
        self.n_mels = C.N_MELS
        self.t_max = C.T_MAX
        self.window = np.hanning(self.n_fft + 1)[:-1]
        self.fb = _mel_filterbank(
            self.sr, self.n_fft, self.n_mels, C.MEL_FMIN, C.MEL_FMAX)

    def _stft_mag(self, x):
        if len(x) < self.n_fft:
            x = np.pad(x, (0, self.n_fft - len(x)))
        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        if n_frames < 1:
            return np.zeros((0, self.n_fft // 2 + 1))
        idx = np.arange(self.n_fft)[None, :] + \
            (np.arange(n_frames) * self.hop)[:, None]
        frames = x[idx] * self.window
        return np.abs(np.fft.rfft(frames, axis=-1))

    def hear(self, wav):
        """波形 → (T_MAX, N_MELS) の対数メル。

        長さを固定するのは①の予測器を単純に保つためで、
        素子＋配線に差し替わる段階では可変長で扱う。
        """
        out = np.full((self.t_max, self.n_mels),
                      np.log(C.LOG_EPS), dtype=np.float64)
        if wav is None or len(wav) == 0:
            return out                      # 沈黙。すべて床値
        mag = self._stft_mag(np.asarray(wav, dtype=np.float64))
        if mag.shape[0] == 0:
            return out
        mel = (mag ** 2) @ self.fb.T
        logmel = np.log(mel + C.LOG_EPS)
        n = min(logmel.shape[0], self.t_max)
        out[:n] = logmel[:n]
        return out

    @property
    def floor(self):
        return float(np.log(C.LOG_EPS))
