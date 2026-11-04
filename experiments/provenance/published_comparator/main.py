import pandas as pd
from scipy.stats import binomtest


def granularity(blocks_df, num):
    '''
    :param blocks_df: dataframe of blocks mined, with
        dates in rows and distinct entities in columns
    :param num: number of days to be combined
    :returns: dataframe of combined blocks mined for the given number of
        days (granularity), with dates in rows and distinct entities in columns
    '''
    daily = blocks_df.T.to_dict('list')
    result = []
    lister = []
    dates = []
    for k, v in sorted(daily.items()):
        lister.append(v)
        dates.append(k)
    i = 0
    indicies = dates
    step = num // 2
    while i <= len(lister)-1:
        combined_blocks = [sum(x) for x in zip(*lister[i-step:i+(num-step):])]
        result.append(combined_blocks)
        i += 1
    new_df = pd.DataFrame(data=result, index=indicies, columns=blocks_df.columns)
    return new_df


def compute_nakamoto_coefficient(row):
    """
    :param row: series of blocks mined by each distinct entity
    :returns: nakamoto coefficient for the given row
    """
    total_blocks = sum(row)
    nc, power_ratio = 0, 0
    if total_blocks > 0:
        for blocks in row.sort_values(ascending=False):
            nc += 1
            power_ratio += blocks / total_blocks
            if power_ratio > 0.5:
                break
    return nc


def compute_nakamoto_coefficients(blocks_df):
    """
    :param blocks_df: dataframe of blocks mined, with
        dates in rows and distinct entities in columns
    :returns: dataframe of nakamoto coefficient, with
        dates in rows and corresponding values of nc in columns
    """
    nc_series = blocks_df.apply(lambda row: compute_nakamoto_coefficient(row), axis=1)
    nc_df = pd.DataFrame( {'nc': nc_series}, index=blocks_df.index)
    return nc_df


def find_nc_range(blocks_df, nc_df, alpha=0.05):
    """
    :param blocks_df: dataframe of blocks mined, with
        dates in rows and distinct entities in columns
    :param nc_df: dataframe of nakamoto coefficient, with
        dates in rows and corresponding values of nc in columns
    :returns: dataframe of range of nakamoto coefficient values, with
        dates in rows and lower, upper nakamoto coefficient in columns
    """
    lower, upper = [], []
    for date in blocks_df.index:
        total_blocks = blocks_df.loc[date].sum(axis=0) 
        coeff = nc_df['nc'].loc[date]
        coeffp, coeffq = coeff, coeff
        if total_blocks > 0:
            sorted_df = blocks_df.loc[date].sort_values(axis=0, ascending=False)
            successes = sorted_df.nlargest(coeff).sum()
            p = binomtest(
                k=successes,
                n=total_blocks,
                p=0.5,
                alternative='greater'
            ).pvalue
            if p > alpha:
                while p > alpha:  # upper
                    coeffp += 1
                    thing = blocks_df.loc[date].sort_values(axis=0, ascending=False)
                    successes = int(thing.nlargest(coeffp).sum())
                    p = binomtest(
                        k=successes,
                        n=total_blocks,
                        p=0.5,
                        alternative='greater'
                    ).pvalue
                coeffp -= 1
            q = binomtest(
                k=successes,
                n=total_blocks,
                p=0.5,
                alternative='less'
            ).pvalue
            if q > alpha:
                while q > alpha:  # lower
                    coeffq -= 1
                    thing = blocks_df.loc[date].sort_values(axis=0, ascending=False)
                    successes = int(thing.nlargest(coeffq).sum())
                    q = binomtest(
                        k=successes,
                        n=total_blocks,
                        p=0.5,
                        alternative='less'
                    ).pvalue
                coeffq += 1
        lower.append(coeffq)
        upper.append(coeffp)
    result = pd.DataFrame({'lower': lower, 'upper': upper}, index=blocks_df.index)
    return result


def binom_p(blocks_df, nc_df, alpha=0.05):
    '''
    Determines the percentage of hypothesis tests passed
    :param blocks_df: dataframe of blocks mined, with
        dates in rows and distinct entities in columns
    :param nc_df: dataframe of nakamoto coefficient, with
        dates in rows and corresponding values of nc in columns
    :returns: percentage of hyptohesis tests passed
    '''
    passes = 0
    total = 0
    for i in range(len(blocks_df.index)):
        num = int(blocks_df.iloc[i].sum(axis=0))
        if num != 0:
            coeff = nc_df['nc'].iloc[i]
            sorted_blocks = blocks_df.iloc[i].sort_values(axis=0, ascending=False)
            successes = int(sorted_blocks.nlargest(coeff).sum())
            p = binomtest(k=successes, n=num, p=0.5, alternative='greater')
            total += 1
            if p.pvalue < alpha:
                passes += 1
    result = (passes/total)*100
    return result


if __name__ == '__main__':
    data_dir = 'data/'
    ledgers = ['bitcoin', 'bitcoin_cash', 'ethereum', 'litecoin', 'zcash']
    dfs = {}

    for ledger in ledgers:
        try:
            df = pd.read_csv(f'{data_dir}/{ledger}_daily.csv', header=0, index_col=0)
        except FileNotFoundError:
            print(f'No data found for {ledger}, so it will be ignored.')
            continue
        df = df.T
        df.index = pd.to_datetime(df.index)
        df = granularity(df, 3)  # for 3-day sliding window
        dfs[ledger] = df
    if 'ethereum' in dfs:
        dfs['ethereum'] = dfs['ethereum'][:'2022-09-14']  # only keep PoW dates for Ethereum
    for ledger, df in dfs.items():
        print(f"Computing Nakamoto Coefficients for {ledger}..")
        nc = compute_nakamoto_coefficients(df)
        tests_passed = binom_p(df, nc)
        print(f'{tests_passed:.2f}% of p-tests passed')
