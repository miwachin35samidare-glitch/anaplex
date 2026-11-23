"""VOICEVOX クライアント。

AnaPlex は AudioQuery を自分で組む（テキストを経由しない）。
`/audio_query` は起動時に音素インベントリを取るためだけに使う。

設計資料 §4.4 の②モーラ層。
"""

import io
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import wave

import numpy as np

import config as C


class VoicevoxError(RuntimeError):
    pass


# 全カナ。1文字ずつ投げて (consonant, vowel) を集める。
# まとめて投げると形態素解析が長い未知語として扱って途中で落ちる
_KANA = (
    "ア イ ウ エ オ カ キ ク ケ コ サ シ ス セ ソ タ チ ツ テ ト "
    "ナ ニ ヌ ネ ノ ハ ヒ フ ヘ ホ マ ミ ム メ モ ヤ ユ ヨ "
    "ラ リ ル レ ロ ワ ヲ ン "
    "ガ ギ グ ゲ ゴ ザ ジ ズ ゼ ゾ ダ ヂ ヅ デ ド バ ビ ブ ベ ボ "
    "パ ピ プ ペ ポ "
    "キャ キュ キョ シャ シュ ショ チャ チュ チョ ニャ ニュ ニョ "
    "ヒャ ヒュ ヒョ ミャ ミュ ミョ リャ リュ リョ "
    "ギャ ギュ ギョ ジャ ジュ ジョ ビャ ビュ ビョ ピャ ピュ ピョ "
    "ッ ヴ ファ フィ フェ フォ ウィ ウェ ウォ "
    "ティ トゥ ディ ドゥ ツァ ツェ ツォ シェ ジェ チェ"
).split()


def _post(url, body=None, timeout=None):
    timeout = timeout or C.VV_TIMEOUT
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    last = None
    for attempt in range(C.VV_RETRY):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last = e
            if attempt < C.VV_RETRY - 1:
                time.sleep(C.VV_RETRY_WAIT)
    raise VoicevoxError(f"{url} に届かない: {last}")


def _get(url, timeout=None):
    timeout = timeout or C.VV_TIMEOUT
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


class Voicevox:
    """凍結された声道。

    バージョンと話者を起動時に照合する。途中で変わると
    (指令 → 音) の対応表が無効になるが、予測誤差は上がるだけなので
    ログからは「探索が再開した」ようにしか見えない（§10）。
    """

    def __init__(self):
        self.url = C.VV_URL.rstrip("/")
        self.speaker = C.VV_SPEAKER_ID
        self.version = None
        self.device = None
        self.inventory = []       # [(consonant|None, vowel), ...] 120種
        self._index = {}          # (c, v) -> id

    # ── 起動 ────────────────────────────────────────────────
    def connect(self):
        self.version = json.loads(_get(f"{self.url}/version"))
        if C.VV_EXPECT_VERSION and self.version != C.VV_EXPECT_VERSION:
            raise VoicevoxError(
                f"ENGINE のバージョンが違う: {self.version} "
                f"(期待 {C.VV_EXPECT_VERSION})。"
                "合成器は凍結対象なので、変わると対応表が無効になる（§10）"
            )
        self.device = json.loads(_get(f"{self.url}/supported_devices"))
        _post(f"{self.url}/initialize_speaker?speaker={self.speaker}")
        return self

    def build_inventory(self):
        """このエンジンで実際に通る (consonant, vowel) を列挙する。

        話者に依存しない（OpenJTalk のテキスト処理から出るため）。
        設計資料 §4.4 の「②で与えるもの: 音素インベントリ」の実体。
        """
        seen = {}
        for kana in _KANA:
            try:
                q = self._audio_query(kana)
            except Exception:
                continue
            for ap in q["accent_phrases"]:
                for m in ap["moras"]:
                    key = (m["consonant"], m["vowel"])
                    seen.setdefault(key, m["text"])
        self.inventory = sorted(seen.keys(), key=lambda k: (k[0] or "", k[1]))
        self._index = {k: i for i, k in enumerate(self.inventory)}
        self.mora_text = [seen[k] for k in self.inventory]
        return self.inventory

    def _audio_query(self, text):
        u = (f"{self.url}/audio_query?speaker={self.speaker}"
             f"&text={urllib.parse.quote(text)}")
        return json.loads(_post(u))

    # ── AudioQuery を自分で組む ─────────────────────────────
    def build_query(self, morae):
        """morae: [(mora_id, consonant_length, vowel_length, pitch), ...]

        空リストなら None を返す（沈黙。VOICEVOX を呼ばない）。
        """
        if not morae:
            return None
        out = []
        for mid, clen, vlen, pitch in morae:
            cons, vowel = self.inventory[mid]
            m = {
                "text": self.mora_text[mid],
                "consonant": cons,
                "consonant_length": float(clen) if cons is not None else None,
                "vowel": vowel,
                "vowel_length": float(vlen),
                "pitch": float(pitch),
            }
            out.append(m)
        return {
            "accent_phrases": [{
                "moras": out,
                "accent": len(out),
                "pause_mora": None,
                "is_interrogative": False,
            }],
            "speedScale": C.VV_SPEED_SCALE,
            "pitchScale": C.VV_PITCH_SCALE,
            "intonationScale": C.VV_INTONATION_SCALE,
            "volumeScale": 1.0,          # 呼び出し側が上書きする
            "prePhonemeLength": C.VV_PRE_PHONEME_LENGTH,
            "postPhonemeLength": C.VV_POST_PHONEME_LENGTH,
            "pauseLength": None,
            "pauseLengthScale": 1.0,
            "outputSamplingRate": C.SAMPLE_RATE,
            "outputStereo": False,
            "kana": "",
        }

    def synthesize(self, query):
        """AudioQuery → 波形(float32, -1..1)。

        スピーカーを経由しない。HTTP レスポンスのバイト列が
        そのまま耳に入る（§6.1）。
        """
        if query is None:
            return np.zeros(0, dtype=np.float32)
        upspeak = str(C.VV_ENABLE_INTERROGATIVE_UPSPEAK).lower()
        u = (f"{self.url}/synthesis?speaker={self.speaker}"
             f"&enable_interrogative_upspeak={upspeak}")
        raw = _post(u, query)
        with wave.open(io.BytesIO(raw), "rb") as w:
            n = w.getnframes()
            pcm = np.frombuffer(w.readframes(n), dtype=np.int16)
        return (pcm.astype(np.float32) / 32768.0)
