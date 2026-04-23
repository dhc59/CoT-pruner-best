"""
句子压缩分析器（基于依存句法）
模拟语文"缩写句子"的方法
"""

import spacy
import logging
from typing import List, Dict, Set, Optional

logger = logging.getLogger(__name__)


class SentenceCompressionAnalyzer:
    """句子压缩分析器（类似语文缩写句子）"""
    
    def __init__(self, spacy_model:  str = "en_core_web_sm"):
        """
        初始化
        
        Args:
            spacy_model: spaCy 模型名称
        """
        try:
            self.nlp = spacy.load(spacy_model)
            logger.info(f"✓ 加载 spaCy 模型: {spacy_model}")
        except OSError:
            logger.warning(f"spaCy 模型 {spacy_model} 未找到，尝试下载...")
            import os
            os.system(f"python -m spacy download {spacy_model}")
            self.nlp = spacy.load(spacy_model)
        
        # 核心依存关系（类似语文中的主谓宾）
        self.core_deps = {
            'ROOT',       # 谓语（句子核心）
            'nsubj',      # 主语
            'nsubjpass',  # 被动语态主语
            'dobj',       # 直接宾语
            'pobj',       # 介词宾语
            'attr',       # 表语
            'ccomp',      # 从句补语
            'xcomp',      # 开放性补语
            'acomp',      # 形容词补语
        }
        
        # 核心词性（实词）
        self.core_pos = {
            'NOUN',       # 名词
            'PROPN',      # 专有名词
            'VERB',       # 动词
            'NUM',        # 数字
            'ADJ',        # 形容词（保留部分）
        }
        
        # 必须保留的词（语义关键词）
        self.must_keep_words = {
            'answer', 'final', 'therefore', 'so', 'because',
            'step', 'first', 'second', 'third', 'last',
            'not', 'no', 'yes',  # 否定/肯定词
            'a', 'b', 'c', 'd', 'e',  # 选项标识
        }
    
    def compress_sentence(self, sentence: str, keep_ratio: float = 0.6) -> str:
        """
        压缩单个句子（保留主干成分）
        
        Args: 
            sentence: 输入句子
            keep_ratio: 保留比例（用于动态调整）
        
        Returns:
            压缩后的句子
        """
        if not sentence. strip():
            return ""
        
        doc = self.nlp(sentence)
        
        # 收集需要保留的 Token 索引
        keep_tokens = set()
        
        for token in doc:
            should_keep = False
            
            # 规则 1: 核心依存关系
            if token. dep_ in self.core_deps:
                should_keep = True
                # 同时保留其核心子节点
                for child in token. children:
                    if child. pos_ in self.core_pos or child.dep_ in self.core_deps:
                        keep_tokens.add(child.i)
            
            # 规则 2: 核心词性
            if token.pos_ in self. core_pos:
                should_keep = True
            
            # 规则 3: 命名实体（人名、地名等）
            if token.ent_type_:
                should_keep = True
            
            # 规则 4: 否定词（非常重要！）
            if token. dep_ == 'neg' or token.text.lower() in {'not', 'no', "n't"}:
                should_keep = True
                # 同时保留被否定的词
                if token.head:
                    keep_tokens.add(token.head. i)
            
            # 规则 5: 关键词
            if token.text.lower() in self.must_keep_words:
                should_keep = True
            
            # 规则 6: 数字和符号
            if token.like_num: 
                should_keep = True
            
            # 规则 7: 句末标点（保持句子完整性）
            if token.is_punct and token.i == len(doc) - 1:
                should_keep = True
            
            # 规则 8: 冒号后的内容（通常是重要信息）
            if token.i > 0 and doc[token.i - 1]. text == ': ':
                should_keep = True
            
            if should_keep:
                keep_tokens.add(token.i)
        
        # 如果保留的 Token 太少，放宽限制
        if len(keep_tokens) < len(doc) * 0.3:
            for token in doc:
                if token.pos_ in {'ADJ', 'ADV'} and token.dep_ in {'amod', 'advmod'}:
                    keep_tokens.add(token.i)
        
        # 按原顺序重构句子
        compressed_tokens = [doc[i]. text for i in sorted(keep_tokens)]
        compressed_text = ' '.join(compressed_tokens)
        
        return compressed_text
    
    def compress_cot(
        self, 
        cot_text:  str, 
        keep_ratio: float = 0.6,
        preserve_structure: bool = True
    ) -> Dict:
        """
        压缩整个 CoT
        
        Args:
            cot_text: 原始 CoT 文本
            keep_ratio: 目标保留比例
            preserve_structure: 是否保留段落结构
        
        Returns:
            压缩结果字典
        """
        if not cot_text.strip():
            return {
                'original_text': cot_text,
                'compressed_text': '',
                'original_tokens': 0,
                'final_tokens': 0,
                'compression_ratio': 0.0,
                'sentences': []
            }
        
        doc = self.nlp(cot_text)
        sentences = list(doc.sents)
        
        compressed_sentences = []
        sentence_details = []
        
        original_tokens_count = 0
        final_tokens_count = 0
        
        for sent in sentences:
            original_sent = sent.text. strip()
            if not original_sent:
                continue
            
            # 压缩句子
            compressed_sent = self.compress_sentence(original_sent, keep_ratio)
            
            if compressed_sent:
                compressed_sentences.append(compressed_sent)
                
                # 记录详情
                original_len = len(list(sent))
                compressed_len = len(self.nlp(compressed_sent))
                
                sentence_details.append({
                    'original':  original_sent,
                    'compressed': compressed_sent,
                    'original_tokens': original_len,
                    'compressed_tokens': compressed_len,
                    'compression':  (original_len - compressed_len) / original_len if original_len > 0 else 0
                })
                
                original_tokens_count += original_len
                final_tokens_count += compressed_len
        
        # 重组文本
        if preserve_structure:
            # 保留原有的换行结构
            compressed_cot = '\n'.join(compressed_sentences) if '\n' in cot_text else ' '.join(compressed_sentences)
        else:
            compressed_cot = ' '.join(compressed_sentences)
        
        # 计算整体压缩率
        compression_ratio = (original_tokens_count - final_tokens_count) / original_tokens_count if original_tokens_count > 0 else 0
        
        return {
            'original_text':  cot_text,
            'compressed_text': compressed_cot,
            'original_tokens':  original_tokens_count,
            'final_tokens': final_tokens_count,
            'compression_ratio': compression_ratio,
            'sentences': sentence_details,
            'num_sentences': len(sentences),
            'num_sentences_kept': len(compressed_sentences),
        }
    
    def compress_steps(self, steps:  List[Dict], keep_ratio: float = 0.6) -> List[Dict]:
        """
        压缩步骤列表（每个步骤单独压缩）
        
        Args:
            steps: 步骤列表，每个步骤是 {'text': str, ... } 格式
            keep_ratio: 保留比例
        
        Returns:
            压缩后的步骤列表
        """
        compressed_steps = []
        
        for step in steps:
            original_text = step['text']
            compressed_text = self. compress_sentence(original_text, keep_ratio)
            
            compressed_step = step.copy()
            compressed_step['text'] = compressed_text
            compressed_step['original_text'] = original_text
            
            compressed_steps.append(compressed_step)
        
        return compressed_steps
