# alt-data-sentiment

> **Languages**: [English](../README.md) · **繁體中文**

我想搞清楚的是：Reddit 情緒到底是新的 cross-sectional factor，還是只是動量的雜訊版本？2021 meme-stock 那波已經很明顯散戶論壇能短期移動股價；難的問題是 —— 控制掉動量跟反轉已經抓到的資訊之後，這效應還剩多少。

## Headline 結果

樣本：leukipp 2021 4 subreddit 語料庫、~55k linked documents、76k `(doc, ticker)` rows、470 檔 S&P 500。

Pooled rank-IC vs forward 21d return 弱（4 個 factor variant 的 HAC |t| 全部 < 2）。有趣的結構在 **per-subreddit**：

| Variant | r/wallstreetbets | r/stocks | r/investing | r/options |
|---|---:|---:|---:|---:|
| `sum_sent_x_attn` | **+2.00** | +0.96 | **−2.41** | +0.15 |
| `shrinkage_sent` | +1.94 | +0.97 | **−3.09** | +0.17 |
| `mean_sent` | +1.79 | +1.08 | **−2.68** | −0.48 |

（HAC t-stats 對 daily rank-IC vs 21d fwd return）

r/investing 是 2021 內統計顯著的反向訊號、WSB 邊際正向。但控制動量 + 反轉後，per-subreddit 訊號完全 collapse（4 個 |t_sent| 全部 < 1.3）。**情緒本質上是 *價格動量的 echo*，不是新的 cross-sectional factor** —— 不過 event study 在 bullish mention spike 上找到真實的短期 meme-momentum 效應（WSB CAR(+10) ≈ +1.52%）。

## 研究問題

散戶論壇談話有雜訊、有偏差、又自利吹捧。但 2021 已經證明它短期會推動股價。我想知道兩件事：扣完成本與 ticker 辨識誤差後，是否還有 *可交易* 的橫斷面情緒因子？它的 alpha 在控制掉 [`classic-factors`](https://github.com/matthiola0/classic-factors) 的動量 / 反轉後，還 *incremental* 嗎？

跨 subreddit 設計是 key check。WSB（meme 群）、r/stocks（一般散戶）、r/investing（長線偏向）、r/options（槓桿偏向）如果訊號真實，行為應該不一樣。若它在 sub-population 之間有可預測的差異，那就是真的訊號；若控制掉因子曝險後就消失，那它就只是動量的偽裝。

## 方法

- **資料**：一個 Kaggle archive — `leukipp/reddit-finance-data`，2021 全年、4 subreddit 的 submissions（WSB、stocks、investing、options）。S&P 500 OHLCV 透過 `qtools.data.loaders.us`。
- **Entity linking**：正則 `\$?[A-Z]{2,5}\b` → S&P 500 白名單 → 常見字 blacklist（CEO / YOLO / FY / …）。4 個 S&P 500 ticker —— `COO`（Cooper）、`DD`（DuPont）、`IT`（Gartner）、`MAR`（Marriott）—— 刻意 blacklist，因為 all-caps 形式被俚語 / 月份壓過；這 4 檔對 pipeline 不可見，已在 `src/alt_sentiment/entity_linking.py` 文件記錄。Notebook 01 對 100 條手標 post 同時驗 precision + recall。
- **情緒模型**：`ProsusAI/finbert`（HuggingFace、CPU batch）。Truncate 到 512 tokens。**每篇文件打一次分**，再 fan-out 到該文件連到的所有 ticker（cost 是 O(documents) 而非 O(documents × tickers)）。
- **日頻因子**：四個變體並列，避免 cherry-pick 友好公式：
  1. `sum((pos − neg) · log1p(mentions))` — 情緒 × 注意力組合
  2. `mean(pos − neg)` — 純情緒
  3. `log1p(doc_count)` — 純注意力
  4. `mean(pos − neg) · n/(n+10)` — 對 thin 名字向 0 收縮
- **評估**：Rank IC 配 **Newey-West / HAC t-stat**（daily 訊號有自相關，純 t-stat 高估顯著性）、p-value、quintile long-short **及 long-only**（月頻、`US_EQUITY` cost）、incremental OLS vs 動量 + 反轉、±10 交易日 mention-spike pump 事件 event study（GameStop 2021-01 是專屬 case study）。
- **跨 subreddit 分析**：除了 pooled 版本，每個 subreddit 各自跑同一因子，使 WSB-vs-rest 對比明確。

## 結果

**Pooled rank-IC**：4 個 factor variant 都沒過 HAC |t| ≥ 2 門檻：

| Variant | Mean rank-IC vs fwd 21d | HAC t-stat (lag 5) |
|---|---:|---:|
| `sum_sent_x_attn` | +0.015 | +1.53 |
| `mean_sent` | +0.012 | +1.51 |
| `attn_log` | −0.020 | −1.07 |
| `shrinkage_sent` | +0.014 | +1.66 |

方向正確（情緒正 → forward return 正），強度不夠。

**Per-subreddit 拆解**（見上方 Headline）。r/investing 的反向訊號可解讀為 *長線散戶看空、後續卻表現好的標的*；r/options 與 r/stocks 基本上是雜訊。

**控制經典因子後**：Per-subreddit pooled HAC OLS 跑 `fwd21 ~ sent + mom + rev`（notebook 03）：

| Subreddit | β_sent | t_sent | n_obs |
|---|---:|---:|---:|
| r/wallstreetbets | −0.000 | −0.22 | 11,202 |
| r/stocks | +0.002 | +1.25 | 9,531 |
| r/investing | +0.002 | +0.30 | 2,462 |
| r/options | +0.001 | +0.18 | 4,438 |

加上動量 + 反轉後，r/investing 的反向訊號消失。情緒在炒作已經在動的標的、動量訊號早就抓到同一份資訊。

**回測**（月頻 rebalance、單邊 5 bps、訊號視窗 2021）：

| Variant | L/S Sharpe | L/S 年化報酬 | LO Sharpe | LO 年化報酬 |
|---|---:|---:|---:|---:|
| `sum_sent_x_attn` | −1.07 | −12.6% | +1.57 | +30.3% |
| `mean_sent` | … | … | **+2.30** | **+45.3%** |
| `attn_log` | … | … | +1.65 | +42.1% |
| `shrinkage_sent` | … | … | +1.69 | +33.2% |

Long-only top-quintile basket 擊敗 2021 SPX（≈ +27%）且 MDD 較小，但與 SPX 高度相關。L/S 那條腿賠錢，因為訊號集中在少數熱門 meme 名字，在強牛年放空底部 quintile 製造大幅負 beta drag。實務解讀：這裡能交易的 edge 在多頭、押在散戶已經堆人的標的 —— 即 follow-through，不是 contrarian alpha。

**Event study**（notebook 04）：139 個 bullish mention spike（mentions ≥ ticker p99 + polarity > 0）平均 abnormal return **+1.17% (t+1) → +1.52% (t+10)**，集中在 WSB（CAR(+5) = +1.43%）vs r/investing（+0.27%）。戰術短期 meme-momentum 效應，不是耐久 cross-sectional factor。

## 結構

```
alt-data-sentiment/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── download_kaggle_wsb.py      # Kaggle archive 拉取
│   ├── download_prices.py          # qtools 抓 S&P 500 OHLCV
│   └── score_sentiment.py          # FinBERT batch scorer
├── notebooks/
│   ├── 01_data_quality.ipynb
│   ├── 02_sentiment_factor.ipynb
│   ├── 03_vs_classic_factors.ipynb
│   └── 04_event_study.ipynb
├── src/alt_sentiment/
│   ├── entity_linking.py           # ticker 抽取 + 白名單 / 黑名單
│   ├── sentiment.py                # FinBERT wrapper
│   ├── factor.py                   # 日頻聚合 + IC + quintile helpers
│   └── loaders/
│       └── leukipp.py
├── reports/figures/
└── data/                           # gitignored
```

## 限制

- **單年樣本**：2021 是 meme-stock 年。跨 subreddit 對比在 2021 內 robust，但對 2022 以後的 external validity 是假設、不是證明。
- **Selection bias**：WSB 是散戶中愛發言的子集，重壓 meme / 高 beta 名字。4-subreddit 設計部分試探這點，但 4 個都偏 bull、皆以美股為主。
- **Ticker 辨識誤差**：黑名單手維護。4 檔真實 S&P 500（`COO`、`DD`、`IT`、`MAR`）刻意排除；其餘 false-positive 表面（如 `NOW`、`LOW` 當強調 vs 公司）保留，由 notebook 01 的 precision AND recall 雙驗。
- **Survivorship / universe freeze**：S&P 500 成分股列表為 `scripts/download_prices.py` 首次執行的 snapshot。後續加入 / 剔除指數的股票一致處理，非歷史精確。與 [`ml-cross-sectional`](https://github.com/matthiola0/ml-cross-sectional)、[`ml-return-forecast`](https://github.com/matthiola0/ml-return-forecast) 同樣的 universe-freeze 簡化。
- **Kaggle archive 而非 live**：本 repo 沒有 forward collector。
- **FinBERT 領域 gap**：在金融新聞訓練、非社群俚語 —— *to the moon 🚀* 這類經常誤判。Notebook 01 對 20 條 WSB 典型句做 spot check。

## Notebook 導覽

- [`01_data_quality.ipynb`](../notebooks/01_data_quality.ipynb) — 原 archive 各 subreddit 覆蓋、entity linking 後 link-rate、top ticker 頻率、naive bot heuristic、FinBERT class probability 分布、10 行文字 spot check。
- [`02_sentiment_factor.ipynb`](../notebooks/02_sentiment_factor.ipynb) — 4 個 factor variant × Newey-West HAC IC × 月頻 L/S × long-only × per-subreddit IC matrix。重點的 subreddit 對比結果在這。
- [`03_vs_classic_factors.ipynb`](../notebooks/03_vs_classic_factors.ipynb) — pairwise correlation、月頻 OLS、pooled HAC regression 跑 fwd21 對 sentiment + 動量 + 反轉。Per-subreddit incremental check。
- [`04_event_study.ipynb`](../notebooks/04_event_study.ipynb) — bullish mention spike 周圍 ±10 交易日 CAR、per subreddit、加單股 case-study panel。

## 重現

```bash
conda create -n alt-data-sentiment python=3.13 -y
conda activate alt-data-sentiment
pip install -e .

# Archive（需要 .env 裡有 KAGGLE_API_TOKEN，見 .env.example）
python scripts/download_kaggle_wsb.py     # leukipp
python scripts/download_prices.py         # 503 檔 S&P 500，2020-2025

# 情緒打分（本地 CPU，leukipp 約 30-60 分鐘）
python scripts/score_sentiment.py --batch-size 16

# 執行 4 個 notebook（scoring 完之後每個 1-3 分鐘）
jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb
```
