# 採掘分散性の再現性資料

論文「What Can Finite Proof-of-Work Block Histories Certify About Mining Decentralization?」の加工済みデータと計算プログラムです。

## 実行方法

実験と `full` の動作確認は **Linux x86_64・Python 3.13.5** で行いました。環境情報は `environment/validated_environment.json`、依存ライブラリは `requirements.txt` に記載しています。

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py replay --output runs/replayed
```

- `verify`：保存済みの参照結果を検証し、論文の数値・図表を再生成します。
- `replay`：加工済みブロックデータと保存済みサンプルから区間・感度分析を再計算します。
- `full`：さらに乱数種からサンプルを再生成し、計算結果を照合します。

出力先は新規または空のディレクトリにします。成功は出力先の `RUN_STATUS.json` の `PASS`、図表・CSV は `publication_assets/` で確認できます。実行ログは出力先の `logs/` に生成されます。依存ライブラリの導入後はオフラインで実行でき、BigQuery 認証や LaTeX は不要です。

## 収録内容

- `empirical/data/`：3チェーンの加工済みデータ、ラベル辞書、データ仕様・ハッシュ。計12,914,458レコード。
- `empirical/code/` と `empirical/design.json`：実データの計算コードと設定。
- `experiments/`：シミュレーションの設定・サンプル・計算コード。
- 両実験の `results` / `results_primary`：再計算結果との照合に必要な参照値。
- `tools/` と `publication_templates/`：図表生成とそのテンプレート。
- `documentation/` と `provenance/`：データの説明、出典、第三者コードとライセンス。

全ファイルの一覧と用途は `CONTENTS.md` を参照してください。`reference_metadata.json` は入力・コードのハッシュと照合用の定数です。報告高さ、履歴度数、候補集合、信頼帯などの派生結果は `replay` で生成します。

数値計算は保存済み採掘者ラベルを前提に再現します。元の BigQuery 取得から最終ラベル付与までの全工程を再構築する資料ではありません。

著者作成コードは MIT、著者作成のデータ・説明資料は CC BY 4.0 です。第三者由来の条件は維持します。引用情報は `CITATION.cff`、ライセンスの適用範囲は `LICENSE_SCOPE.md` にあります。
