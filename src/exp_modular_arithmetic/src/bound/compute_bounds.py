import numpy as np
from scipy import optimize
import scipy
from sklearn.cluster import KMeans
from decimal import getcontext
from decimal import Decimal

def get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype):
    kmeans = KMeans(n_clusters=levels).fit(vec.reshape(-1, 1))
    codebook = kmeans.cluster_centers_.astype(codebook_dtype)[:, 0]
    symbols = kmeans.labels_
    return symbols, codebook

def get_random_symbols_and_codebook(vec, levels, codebook_dtype):
    largest = max(np.max(vec), np.abs(np.min(vec)))
    initvals = np.linspace(-largest - 1e-6, largest + 1e-6, levels + 1)
    assignments = np.digitize(vec, initvals) - 1
    centroids = []
    for i in range(levels):
        aux = vec[assignments == i]
        if len(aux) > 0:
            centroids.append(np.mean(aux))
        else:
            centroids.append(initvals[i])
    codebook = np.array(centroids, dtype=codebook_dtype)
    symbols = np.array(assignments)
    return symbols, codebook

def decimal2bits(decimal, bits_encoded):
    output_bits = []
    while len(output_bits) < bits_encoded:
        if decimal > Decimal(1) / Decimal(2):
            output_bits.append(1)
            decimal -= Decimal(1) / Decimal(2)
        else:
            output_bits.append(0)
        decimal *= Decimal(2)
    return output_bits

def encode(sequence, probs):
    """Arithmetic coding of sequence of integers Seq: [a0,a1,a2,...]
    with probabilities: [c0,c1,c2,...]"""
    cumulative_probs = np.cumsum(probs)
    width = Decimal(1)
    message_value = Decimal(0)
    bits_encoded = 0
    for i, val in enumerate(sequence):
        bin_start = cumulative_probs[val - 1] if val > 0 else 0.0
        bin_size = probs[val]
        message_value = message_value + Decimal(bin_start) * width
        width = width * Decimal(bin_size)
        bits_encoded -= np.log2(bin_size)
    print(f"arithmetic encoded bits {bits_encoded:.2f}")
    return decimal2bits(message_value + width / 2, np.ceil(bits_encoded) + 1)

def do_arithmetic_encoding(symbols, probabilities, levels):
    entropy_est = scipy.stats.entropy(probabilities, base=2)
    is_too_large_to_run = len(symbols) > int(1e4)
    if is_too_large_to_run:
        coded_symbols_size = np.ceil(len(symbols) * entropy_est) + 1.
    else:
        getcontext().prec = int(1.1 * np.log10(levels) * len(symbols))
        coded_symbols_size = len(encode(symbols, probabilities))
    return symbols, coded_symbols_size

def get_message_len(coded_symbols_size, codebook, max_count):
    codebook_bits_size = 16 if codebook.dtype == np.float16 else 32
    probability_bits = int(np.ceil(np.log2(max_count)) * len(codebook))
    codebook_bits = len(codebook) * codebook_bits_size
    summary = f"encoding {coded_symbols_size}, codebook {codebook_bits} probs {probability_bits}"
    print(summary)
    message_len = coded_symbols_size + codebook_bits + probability_bits
    return message_len

def quantize_vector(vec):
    # 固定参数
    levels = 2**2 + 1  # 这里为 5
    use_kmeans = False
    encoding_type = "arithmetic"
    codebook_dtype = np.float32  # 固定为 float32

    # 生成符号和码本
    if use_kmeans:
        symbols, codebook = get_kmeans_symbols_and_codebook(vec, levels, codebook_dtype)
    else:
        symbols, codebook = get_random_symbols_and_codebook(vec, levels, codebook_dtype)
    
    # 计算每个符号出现的概率
    probabilities = np.array([np.mean(symbols == i) for i in range(levels)])

    # 根据选择的编码算法对符号进行编码
    if encoding_type == "arithmetic":
        _, coded_symbols_size = do_arithmetic_encoding(symbols, probabilities, levels)
    #elif encoding_type == "huff":
    #    _, coded_symbols_size = do_huffman_encoding(symbols)
    else:
        raise NotImplementedError("Encoding type not implemented")

    # 通过码本重构向量
    decoded_vec = np.zeros(shape=(len(vec)))
    for k in range(len(codebook)):
        decoded_vec[symbols == k] = codebook[k]

    # 计算描述长度（message len）
    message_len = get_message_len(coded_symbols_size, codebook, len(symbols))
    print(f"Message Len: {message_len}")
    
    return decoded_vec, message_len


def llm_subsampling_bound(train_error, div, data_size, sample_size, delta = 1, epsilon=0.05):
    r = sample_size/(sample_size + data_size)
    complexity = np.sqrt((div - np.log(r * epsilon))/(2*data_size)) +np.sqrt(-np.log((1-r) * epsilon)/(2*sample_size))
    bound = train_error
    print("delta: ", delta)
    print("compl: ", complexity)
    return bound+delta*complexity