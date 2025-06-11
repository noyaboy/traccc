import pandas as pd

def clean_gru_data(input_path: str,
                   output_path: str,
                   expected_fields: int = 70) -> None:
    """
    1) 只讀剛好有 expected_fields 欄的列（其他自動跳過）
    2) 去除完全重複的列
    3) 存檔
    """
    # names=range(...) 強制告訴 pandas 預期欄位數
    df = pd.read_csv(
        input_path,
        header=None,
        names=range(expected_fields),
        dtype=str,
        engine='python',
        sep=',',
        on_bad_lines='skip'   # pandas ≥ 1.3.0，跳過欄位數不對的列
    )

    # 去重
    df = df.drop_duplicates(keep='first')

    # 輸出
    df.to_csv(output_path, index=False, header=False)

if __name__ == "__main__":
    clean_gru_data("gru_training_data.csv", "gru_training_data_clean.csv")
    print("清洗完成，結果存到 gru_training_data_clean.csv")
