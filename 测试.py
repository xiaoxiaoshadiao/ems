







# sea,eat

def zichuan(s1, s2):
    m, n = len(s1), len(s2)
    if m == 0 or n == 0:
        return ''

    dp = [[0]*(n+1) for _ in range(m+1)]
    max_len = 0
    end_index = 0

    for i in range(1, m+1):
        for j in range(1, n+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1

                if dp[i][j] > max_len:
                    max_len = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0
    if max_len == 0:
        return ''
    return s1[end_index - max_len: end_index]

if __name__ == '__main__':
    parts = input().strip().split(',')
    s1 = parts[0]
    s2 = parts[1]
    # print(s1)
    # print(s2)
    print(zichuan(s1, s2))