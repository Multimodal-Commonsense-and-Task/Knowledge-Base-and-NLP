# -*- coding: utf-8 -*-
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ROUGe metric implementation.
This code is from
https://github.com/google/seq2seq/blob/master/seq2seq/metrics/rouge.py
This is a modified and slightly extended verison of
https://github.com/miso-belica/sumy/blob/dev/sumy/evaluation/rouge.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import itertools
import numpy as np

#pylint: disable=C0103

def _get_ngrams(n, text):
  """Calcualtes n-grams.

  Args:
    n: which n-grams to calculate
    text: An array of tokens

  Returns:
    A set of n-grams
  """
  ngram_set = set()
  text_length = len(text)
  max_index_ngram_start = text_length - n
  for i in range(max_index_ngram_start + 1):
    ngram_set.add(tuple(text[i:i + n]))
  return ngram_set


def _split_into_words(sentences):
  """Splits multiple sentences into words and flattens the result"""
  return list(itertools.chain(*[_.split(" ") for _ in sentences]))


def _get_word_ngrams(n, sentences):
  """Calculates word n-grams for multiple sentences.
  """
  assert len(sentences) > 0
  assert n > 0

  words = _split_into_words(sentences)
  return _get_ngrams(n, words)


def _len_lcs(x, y):
  """
  Returns the length of the Longest Common Subsequence between sequences x
  and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns
    integer: Length of LCS between x and y
  """
  table = _lcs(x, y)
  n, m = len(x), len(y)
  return table[n, m]


def _lcs(x, y):
  """
  Computes the length of the longest common subsequence (lcs) between two
  strings. The implementation below uses a DP programming algorithm and runs
  in O(nm) time where n = len(x) and m = len(y).
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: collection of words
    y: collection of words

  Returns:
    Table of dictionary of coord and len lcs
  """
  n, m = len(x), len(y)
  table = dict()
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 or j == 0:
        table[i, j] = 0
      elif x[i - 1] == y[j - 1]:
        table[i, j] = table[i - 1, j - 1] + 1
      else:
        table[i, j] = max(table[i - 1, j], table[i, j - 1])
  return table


def _recon_lcs(x, y):
  """
  Returns the Longest Subsequence between x and y.
  Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence

  Args:
    x: sequence of words
    y: sequence of words

  Returns:
    sequence: LCS of x and y
  """
  i, j = len(x), len(y)
  table = _lcs(x, y)

  def _recon(i, j):
    """private recon calculation"""
    if i == 0 or j == 0:
      return []
    elif x[i - 1] == y[j - 1]:
      return _recon(i - 1, j - 1) + [(x[i - 1], i)]
    elif table[i - 1, j] > table[i, j - 1]:
      return _recon(i - 1, j)
    else:
      return _recon(i, j - 1)

  recon_tuple = tuple(map(lambda x: x[0], _recon(i, j)))
  return recon_tuple


def rouge_n(evaluated_sentences, reference_sentences, n=2):
  """
  Computes ROUGE-N of two text collections of sentences.
  Sourece: http://research.microsoft.com/en-us/um/people/cyl/download/
  papers/rouge-working-note-v1.3.1.pdf

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set
    n: Size of ngram.  Defaults to 2.

  Returns:
    A tuple (f1, precision, recall) for ROUGE-N

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  evaluated_ngrams = _get_word_ngrams(n, evaluated_sentences)
  reference_ngrams = _get_word_ngrams(n, reference_sentences)
  reference_count = len(reference_ngrams)
  evaluated_count = len(evaluated_ngrams)

  # Gets the overlapping ngrams between evaluated and reference
  overlapping_ngrams = evaluated_ngrams.intersection(reference_ngrams)
  overlapping_count = len(overlapping_ngrams)

  # Handle edge case. This isn't mathematically correct, but it's good enough
  if evaluated_count == 0:
    precision = 0.0
  else:
    precision = overlapping_count / evaluated_count

  if reference_count == 0:
    recall = 0.0
  else:
    recall = overlapping_count / reference_count

  f1_score = 2.0 * ((precision * recall) / (precision + recall + 1e-8))

  # return overlapping_count / reference_count
  return f1_score, precision, recall


def _f_p_r_lcs(llcs, m, n):
  """
  Computes the LCS-based F-measure score
  Source: http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Args:
    llcs: Length of LCS
    m: number of words in reference summary
    n: number of words in candidate summary

  Returns:
    Float. LCS-based F-measure score
  """
  r_lcs = llcs / m
  p_lcs = llcs / n
  beta = p_lcs / (r_lcs + 1e-12)
  num = (1 + (beta**2)) * r_lcs * p_lcs
  denom = r_lcs + ((beta**2) * p_lcs)
  f_lcs = num / (denom + 1e-12)
  return f_lcs, p_lcs, r_lcs


def rouge_l_sentence_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (sentence level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = LCS(X,Y)/m
  P_lcs = LCS(X,Y)/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  X = reference summary
  Y = Candidate summary
  m = length of reference summary
  n = length of candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentences: The sentences from the referene set

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")
  reference_words = _split_into_words(reference_sentences)
  evaluated_words = _split_into_words(evaluated_sentences)
  m = len(reference_words)
  n = len(evaluated_words)
  lcs = _len_lcs(evaluated_words, reference_words)
  return _f_p_r_lcs(lcs, m, n)


def _union_lcs(evaluated_sentences, reference_sentence):
  """
  Returns LCS_u(r_i, C) which is the LCS score of the union longest common
  subsequence between reference sentence ri and candidate summary C. For example
  if r_i= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and
  c2 = w1 w3 w8 w9 w5, then the longest common subsequence of r_i and c1 is
  “w1 w2” and the longest common subsequence of r_i and c2 is “w1 w3 w5”. The
  union longest common subsequence of r_i, c1, and c2 is “w1 w2 w3 w5” and
  LCS_u(r_i, C) = 4/5.

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    float: LCS_u(r_i, C)

  ValueError:
    Raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  lcs_union = set()
  reference_words = _split_into_words([reference_sentence])
  combined_lcs_length = 0
  for eval_s in evaluated_sentences:
    evaluated_words = _split_into_words([eval_s])
    lcs = set(_recon_lcs(reference_words, evaluated_words))
    combined_lcs_length += len(lcs)
    lcs_union = lcs_union.union(lcs)

  union_lcs_count = len(lcs_union)
  union_lcs_value = union_lcs_count / combined_lcs_length
  return union_lcs_value


def rouge_l_summary_level(evaluated_sentences, reference_sentences):
  """
  Computes ROUGE-L (summary level) of two text collections of sentences.
  http://research.microsoft.com/en-us/um/people/cyl/download/papers/
  rouge-working-note-v1.3.1.pdf

  Calculated according to:
  R_lcs = SUM(1, u)[LCS<union>(r_i,C)]/m
  P_lcs = SUM(1, u)[LCS<union>(r_i,C)]/n
  F_lcs = ((1 + beta^2)*R_lcs*P_lcs) / (R_lcs + (beta^2) * P_lcs)

  where:
  SUM(i,u) = SUM from i through u
  u = number of sentences in reference summary
  C = Candidate summary made up of v sentences
  m = number of words in reference summary
  n = number of words in candidate summary

  Args:
    evaluated_sentences: The sentences that have been picked by the summarizer
    reference_sentence: One of the sentences in the reference summaries

  Returns:
    A float: F_lcs

  Raises:
    ValueError: raises exception if a param has len <= 0
  """
  if len(evaluated_sentences) <= 0 or len(reference_sentences) <= 0:
    raise ValueError("Collections must contain at least 1 sentence.")

  # total number of words in reference sentences
  m = len(_split_into_words(reference_sentences))

  # total number of words in evaluated sentences
  n = len(_split_into_words(evaluated_sentences))

  union_lcs_sum_across_all_references = 0
  for ref_s in reference_sentences:
    union_lcs_sum_across_all_references += _union_lcs(evaluated_sentences,
                                                      ref_s)
  return _f_p_r_lcs(union_lcs_sum_across_all_references, m, n)


def rouge(hypotheses, references):
  """Calculates average rouge scores for a list of hypotheses and
  references"""

  # Filter out hyps that are of 0 length
  # hyps_and_refs = zip(hypotheses, references)
  # hyps_and_refs = [_ for _ in hyps_and_refs if len(_[0]) > 0]
  # hypotheses, references = zip(*hyps_and_refs)

  # Calculate ROUGE-1 F1, precision, recall scores
  rouge_1 = [
      rouge_n([hyp], [ref], 1) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_1_f, rouge_1_p, rouge_1_r = map(np.mean, zip(*rouge_1))

  # Calculate ROUGE-2 F1, precision, recall scores
  rouge_2 = [
      rouge_n([hyp], [ref], 2) for hyp, ref in zip(hypotheses, references)
  ]
  rouge_2_f, rouge_2_p, rouge_2_r = map(np.mean, zip(*rouge_2))

  # Calculate ROUGE-L F1, precision, recall scores
  rouge_l = [
      rouge_l_sentence_level([hyp], [ref])
      for hyp, ref in zip(hypotheses, references)
  ]
  rouge_l_f, rouge_l_p, rouge_l_r = map(np.mean, zip(*rouge_l))

  return {
      "rouge_1/f": rouge_1_f,
      "rouge_1/r": rouge_1_r,
      "rouge_1/p": rouge_1_p,
      "rouge_2/f": rouge_2_f,
      "rouge_2/r": rouge_2_r,
      "rouge_2/p": rouge_2_p,
      "rouge_l/f": rouge_l_f,
      "rouge_l/r": rouge_l_r,
      "rouge_l/p": rouge_l_p,
  }

if __name__ == "__main__":
  from konlpy.tag import Okt
  okt = Okt()
  # [o for o in output if o[1]!="Josa"]
  # Test the functions
  evaluated_sentences = [ #In korean
      "제천 화재의 원인은 여러 가지 요인들이 복합적으로 작용한 결과로 보입니다. \n\n1.  **보온등 과열**: 국립과학수사연구원은 1층 주차장 천장의 보온등이 축열(과열)로 인해 화재가 발생한 것으로 추정했다. 보온등은 폐수관 동파를 막기 위해 설치되었지만, 관리 소홀으로 인해 과열을 일으켜 화재를 유발했다.\n2.  **건물 구조**: 건물은 필로티 구조로 되어 있어, 1층 주차장에서 시작된 불이 빠르게 상층부로 옮겨졌습니다. 또한, 건물의 외장재는 드라이비트로 되어 있어, 유독 가스를 다량 발생시켜 화재를 키웠습니다.\n3.  **소방 대응 부실**: 소방 당국은 초기 대응 부실로 인명 피해를 키웠습니다. 2층 여성 사우나 구조 지연에 대해 현장 지휘관들의 책임이 크다고 봤습니다. 충북 119상황실에서 2층에 다수의 요구조자가 있다는 사실을 3차례나 현장에 전달했지만, 지휘 책임자들은 현장대원들에게 즉각적으로 상황을 전달하거나 구조 지시를 내리지 않았다.\n4.  **건물 관리 소홀**: 건물주의 소방안전관리 소홀과 화재에 취약한 건물구조상의 문제도 지적되었다. 화재 건물은 4·5·7층에 설치된 10개의 배연창이 잠겨 있었고, 스프링클러 알람밸브와 보조펌프 개폐밸브도 폐쇄돼 있었다. 화재감지기가 잘못 시공되고 방화셔터도 작동하지 않는 등 안전시설 관리가 법령을 위반했다고 보고 경찰에 수사의뢰키로 했다.\n\n이러한 요인들이 복합적으로 작용하여 제천 화재가 발생한 것으로 보입니다.",
  ]
  evaluated_sentences = okt.pos(evaluated_sentences[0])
  evaluated_sentences = [o[0] for o in evaluated_sentences if o[1] != "Josa"]
  print(evaluated_sentences)
  evaluated_sentences = [" ".join([o for o in evaluated_sentences])]
  print(evaluated_sentences)
  reference_sentences = [
    "제천 화재는 2017년 12월 21일 충청북도 제천시의 스포츠센터에서 발생한 화재로, 29명이 사망하고 36명이 부상당하는 등 큰 인명피해를 냈습니다.\n\n화재의 원인은 전기적 요인으로 추정되고 있습니다. 정확한 원인은 조사 결과에 따르면, 스포츠센터의 지하 1층에 있는 주차장의 전기실에서 화재가 발생한 것으로 밝혀졌습니다. 전기실에서 사용되는 전기기기의 과열로 인해 화재가 발생한 것으로 추정됩니다.\n\n이 화재는 스포츠센터의 화재 안전 대책이 미흡했던 점도 문제로 지적되었습니다. 스포츠센터의 소방 시설이 제대로 작동하지 않았고, 화재 경보 시스템도 정상적으로 작동하지 않았던 것으로 밝혀졌습니다.\n\n제천 화재는 화재 안전의 중요성을 다시 한번 강조하는 계기가 되었습니다. 건물의 화재 안전 대책을 철저히 하여, 이러한 비극적인 사고가 다시 발생하지 않도록 해야 합니다.\n\n이러한 사고를 방지하기 위해서는 건물의 화재 안전 대책을 철저히 하고, 소방 시설을 정기적으로 점검하며, 화재 안전 교육을 실시하는 등 다양한 노력을 기울여야 합니다."
  ]
  reference_sentences = okt.pos(reference_sentences[0])
  reference_sentences = [o[0] for o in reference_sentences if o[1] != "Josa"]
  reference_sentences = [" ".join([o for o in reference_sentences])]
  print(evaluated_sentences, reference_sentences)
  print(rouge(evaluated_sentences, reference_sentences))