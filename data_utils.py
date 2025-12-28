import json
import os
import torch
from torch.utils.data import Dataset
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranslationDataset(Dataset):
    """
    自定义翻译数据集加载器，支持 JSONL 格式读取与基础清洗。
    每行期望为 {"zh": "...", "en": "..."}
    """
    def __init__(self, filepath, src_tokenizer, tgt_tokenizer, max_len=512, min_len=2):
        self.data = []
        self.src_tokenizer = src_tokenizer
        self.tgt_tokenizer = tgt_tokenizer
        self.max_len = max_len
        self.min_len = min_len

        assert os.path.exists(filepath), f"JSONL 文件未找到: {filepath}"
        logger.info(f"Loading data from {filepath}...")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    pair = json.loads(line)
                    if self._validate_pair(pair):
                        self.data.append(pair)
                except json.JSONDecodeError:
                    # 跳过损坏行
                    continue
        logger.info(f"Loaded {len(self.data)} valid sentence pairs.")

    def _validate_pair(self, pair):
        """
        验证句对的有效性。
        过滤逻辑：
        1. 包含源和目标键值。
        2. 长度在阈值范围内。
        3. 长度比（Length Ratio）不过于悬殊（例如1:5或5:1），防止对齐错误的数据干扰训练。
        """
        src = pair.get('zh', '').strip()
        tgt = pair.get('en', '').strip()
        
        if not src or not tgt:
            return False
            
        if len(src) > self.max_len or len(tgt) > self.max_len:
            return False
            
        if len(src) < self.min_len or len(tgt) < self.min_len:
            return False
            
        # 启发式规则：长度比例过滤
        # 中文通常比英文短（字符数vs单词数），但差异不应超过特定倍数
        ratio = max(1e-6, len(src)) / max(1, len(tgt.split()))
        if ratio < 0.2 or ratio > 5.0: # 宽松的界限
            return False
            
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回原始文本，分词与数值化将在Collate Function中批量处理以提高效率
        return self.data[idx]

from collections import Counter
import jieba

# 英文分词优先使用 SacreMoses；仅在缺失时回退到 NLTK，再不行用空格。
_HAS_SACREMOSES = False
_HAS_NLTK = False

try:
    from sacremoses import MosesTokenizer
    _HAS_SACREMOSES = True
except Exception:
    _HAS_SACREMOSES = False

if not _HAS_SACREMOSES:
    try:
        import nltk
        _HAS_NLTK = True
        # 优先使用本地 nltk_data（包含 punkt_tab/punkt），避免联网下载
        NLTK_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'nltk_data'))
        if os.path.isdir(NLTK_DATA_DIR):
            nltk.data.path.insert(0, NLTK_DATA_DIR)
        # 确保 punkt_tab 优先可用
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            try:
                nltk.download('punkt_tab', download_dir=NLTK_DATA_DIR)
            except Exception:
                pass
        # word_tokenize 仍依赖 punkt，若缺失则补齐到本地目录
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            try:
                nltk.download('punkt', download_dir=NLTK_DATA_DIR)
            except Exception:
                pass
    except Exception:
        _HAS_NLTK = False

# 可选：SentencePiece 子词分词支持
_HAS_SENTENCEPIECE = False
try:
    import sentencepiece as spm
    _HAS_SENTENCEPIECE = True
except Exception:
    _HAS_SENTENCEPIECE = False

class Vocabulary:
    def __init__(self, freq_threshold=2, lang='en'):
        self.itos = {0: "<pad>", 1: "<sos>", 2: "<eos>", 3: "<unk>"}
        self.stoi = {tok: idx for idx, tok in self.itos.items()}
        self.freq_threshold = freq_threshold
        self.lang = lang

        # 初始化英文分词器
        if self.lang == 'en':
            if _HAS_SACREMOSES:
                self._mtok = MosesTokenizer(lang='en')
            else:
                self._mtok = None

    def __len__(self):
        return len(self.itos)

    def tokenizer(self, text: str):
        text = text.strip()
        if self.lang == 'zh':
            return [tok for tok in jieba.cut(text) if tok]
        # 英文优先 Moses → NLTK → 空格
        if _HAS_SACREMOSES and self._mtok is not None:
            return self._mtok.tokenize(text, return_str=False)
        if _HAS_NLTK:
            return nltk.word_tokenize(text.lower())
        return text.lower().split()

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4  # 从 4 开始，预留特殊 token

        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            frequencies.update(tokens)

        for word, count in frequencies.items():
            if count >= self.freq_threshold and word not in self.stoi:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        tokens = self.tokenizer(text)
        return [self.stoi.get(token, self.stoi["<unk>"]) for token in tokens]


class SubwordVocab:
    """
    使用 SentencePiece 的子词词表包装，要求训练 spm.model 时设置特殊ID：
    pad_id=0, bos_id=1, eos_id=2, unk_id=3。
    """
    def __init__(self, spm_model_path: str, lang: str = 'en'):
        assert _HAS_SENTENCEPIECE, "sentencepiece 未安装，请先 pip install sentencepiece"
        assert os.path.exists(spm_model_path), f"SPM 模型未找到: {spm_model_path}"
        self.lang = lang
        self.spp = spm.SentencePieceProcessor()
        self.spp.load(spm_model_path)

        # 约定：训练 SPM 时使用 pad_id=0, bos_id=1, eos_id=2, unk_id=3
        # 这里显式记录这些索引，并保证 stoi 中存在对应的字符串键
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

        # 构建 itos：先写入特殊 token，然后覆盖/补充为 SPM 的 piece 名称
        self.itos = {
            self.pad_idx: "<pad>",
            self.sos_idx: "<sos>",
            self.eos_idx: "<eos>",
            self.unk_idx: "<unk>",
        }
        for i in range(self.spp.get_piece_size()):
            piece = self.spp.id_to_piece(i)
            self.itos[i] = piece

        # 构建 stoi，并确保自定义特殊 token 字符串可用
        self.stoi = {tok: idx for idx, tok in self.itos.items()}
        self.stoi["<pad>"] = self.pad_idx
        self.stoi["<sos>"] = self.sos_idx
        self.stoi["<eos>"] = self.eos_idx
        self.stoi["<unk>"] = self.unk_idx

    def __len__(self):
        return max(self.itos.keys()) + 1

    def tokenizer(self, text: str):
        return self.spp.encode(text.strip(), out_type=int)

    def build_vocabulary(self, sentence_list):
        # 子词词表由 SPM 提供，无需统计
        return

    def numericalize(self, text):
        return self.tokenizer(text)

# 批处理辅助函数 (Collate Function)
from torch.nn.utils.rnn import pad_sequence

class Collate:
    """
    组装批次：返回
    - src: [B, S]
    - src_mask: [B, S] (pad=0, 非pad=1)
    - tgt_inp: [B, T] (<sos> + y)
    - tgt_out: [B, T] (y + <eos>)
    """
    def __init__(self, src_vocab, tgt_vocab, device):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.device = device

    def __call__(self, batch):
        src_batch, tgt_inp_batch, tgt_out_batch = [], [], []

        for item in batch:
            src_tokens = self.src_vocab.numericalize(item['zh'])
            tgt_tokens = self.tgt_vocab.numericalize(item['en'])

            src_indices = [self.src_vocab.stoi["<sos>"]] + src_tokens + [self.src_vocab.stoi["<eos>"]]
            tgt_inp = [self.tgt_vocab.stoi["<sos>"]] + tgt_tokens
            tgt_out = tgt_tokens + [self.tgt_vocab.stoi["<eos>"]]

            src_batch.append(torch.tensor(src_indices, dtype=torch.long))
            tgt_inp_batch.append(torch.tensor(tgt_inp, dtype=torch.long))
            tgt_out_batch.append(torch.tensor(tgt_out, dtype=torch.long))

        src = pad_sequence(src_batch, padding_value=self.src_vocab.stoi["<pad>"], batch_first=True)
        tgt_inp = pad_sequence(tgt_inp_batch, padding_value=self.tgt_vocab.stoi["<pad>"], batch_first=True)
        tgt_out = pad_sequence(tgt_out_batch, padding_value=self.tgt_vocab.stoi["<pad>"], batch_first=True)

        src_mask = (src != self.src_vocab.stoi["<pad>"]).long()

        return src.to(self.device), src_mask.to(self.device), tgt_inp.to(self.device), tgt_out.to(self.device)