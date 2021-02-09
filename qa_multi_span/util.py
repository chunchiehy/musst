import re


def untokenize(words):
  """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
  text = ' '.join(words)
  step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .', '...')
  step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
  step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
  step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
  step5 = step4.replace(" '", "'").replace(" n't",
                                           "n't").replace("can not", "cannot")
  step6 = step5.replace(" ` ", " '")
  step7 = step6.replace("$ ", "$")
  return step7.strip()


# KMP Algorithm
def KMPSearch(pat, txt):
  index = -1
  M = len(pat)
  N = len(txt)
  # create lps[] that will hold the longest prefix suffix
  # values for pattern
  lps = [0] * M
  j = 0  # index for pat[]
  # Preprocess the pattern (calculate lps[] array)
  computeLPSArray(pat, M, lps)
  i = 0  # index for txt[]
  while i < N:
    if pat[j] == txt[i]:
      i += 1
      j += 1
    if j == M:
      #  print ("Found pattern at index " + str(i-j))
      index = i - j
      j = lps[j - 1]
      break
    # mismatch after j matches
    elif i < N and pat[j] != txt[i]:
      # Do not match lps[0..lps[j-1]] characters,
      # they will match anyway
      if j != 0:
        j = lps[j - 1]
      else:
        i += 1
  return index


def computeLPSArray(pat, M, lps):
  len = 0  # length of the previous longest prefix suffix
  lps[0]  # lps[0] is always 0
  i = 1
  # the loop calculates lps[i] for i = 1 to M-1
  while i < M:
    if pat[i] == pat[len]:
      len += 1
      lps[i] = len
      i += 1
    else:
      # This is tricky. Consider the example.
      # AAACAAAA and i = 7. The idea is similar
      # to search step.
      if len != 0:
        len = lps[len - 1]
        # Also, note that we do not increment i here
      else:
        lps[i] = 0
        i += 1


def encode_pq(tokenizer, query, passage, max_len, task='nlg'):
  # Doing some encoding to get [ans_tokens, qp_text_tokens]
  if task == 'qa':
    passage =  passage + ' yes, no, it is'
  
  q_tokens = tokenizer.tokenize(query)
  p_tokens = tokenizer.tokenize(passage)
  max_p_len = max_len - len(q_tokens) - 3
  if len(p_tokens) > max_p_len:
    p_tokens = p_tokens[:max_p_len]
  qp_text_tokens = [tokenizer.cls_token] + q_tokens + [
      tokenizer.sep_token
  ] + p_tokens + [tokenizer.sep_token]

  return qp_text_tokens, p_tokens, q_tokens