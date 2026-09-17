import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os


def extract_language(dataset: str, part: str) -> str:
    """Extract language from the 'dataset' and 'part' columns."""
    dataset_str = str(dataset).lower() if dataset is not None else ""
    part_str = str(part) if part is not None else ""

    # 1. Check for explicit Math / Code in both dataset and part
    if "math" in dataset_str or "math" in part_str.lower():
        return "Math"
    if (
        "code" in dataset_str
        or "code" in part_str.lower()
        or "starcoder" in dataset_str
    ):
        return "Code"

    # 2. Extract specific language tags
    tags = re.findall(r"[a-z]{3}_[A-Z][a-z]{3}", part_str)

    if tags:
        if len(tags) == 1:
            return tags[0]
        # If multiple tags, pick the non-English one
        for tag in tags:
            if tag != "eng_Latn":
                return tag
        return tags[0]

    # 3. Default to English if no tag is found and not math/code
    return "eng_Latn"


def get_source_type(dataset: str) -> str:
    """Categorize dataset into source types."""
    ds = str(dataset).lower()

    if any(ds.startswith(prefix) for prefix in ["hplt", "dclm", "nemotron-cc-1.0"]):
        return "Web Data"

    if any(
        prefix in ds
        for prefix in [
            "starcoder",
            "megamath",
            "swallow",
            "finemath",
            "dolmino-mix",
            "openwebmath",
            "common-pile",
        ]
    ):
        return "Code & Math"

    if any(
        prefix in ds for prefix in ["finewiki", "finepdfs", "finephrase", "agenttrove"]
    ):
        return "Curated (Wiki/PDF/Books)"

    if any(
        prefix in ds
        for prefix in [
            "nemotron-cc-tower",
            "nemotron-cc-opus",
            "fineopus",
            "dochplt",
            "mixture-vitae",
            "nemotron-pretraining",
            "nemotron-mind",
        ]
    ):
        return "Translations / Synthetic"

    return "Other Sources"


def analyze_and_visualize(csv_path: str, assets_dir: str):
    """Load data, analyze languages and sources, and create visualizations."""
    os.makedirs(assets_dir, exist_ok=True)
    df = pl.read_csv(csv_path)

    # Ensure budget is numeric
    df = df.with_columns(pl.col("budget").cast(pl.Float64))

    # Process dataset for language and source_type
    df = df.with_columns(
        pl.struct(["dataset", "part"])
        .map_elements(
            lambda x: extract_language(x["dataset"], x["part"]), return_dtype=pl.Utf8
        )
        .alias("language"),
        pl.col("dataset")
        .map_elements(get_source_type, return_dtype=pl.Utf8)
        .alias("source_type"),
    )

    # --- Language Analysis ---
    agg_lang = (
        df.group_by("language")
        .agg(pl.col("budget").sum().alias("total_tokens"))
        .sort("total_tokens", descending=True)
    )

    total_budget = agg_lang["total_tokens"].sum()
    agg_lang = agg_lang.with_columns(
        (pl.col("total_tokens") / total_budget * 100).alias("percentage")
    )
    print("Top Languages:")
    print(agg_lang.head(10))
    agg_lang.write_csv(f"{assets_dir}/aggregated_tokens.csv")

    # Visualizations
    sns.set_theme(style="whitegrid")

    # 1. Bar Chart for Languages
    plt.figure(figsize=(12, 10))
    pd_lang = agg_lang.to_pandas()
    sns.barplot(
        data=pd_lang,
        x="percentage",
        y="language",
        hue="language",
        palette="viridis",
        legend=False,
    )
    plt.title("Tokens by Language (%) - OpenEuroLLM Flagship Mix")
    plt.xlabel("Percentage of Total Tokens (%)")
    plt.ylabel("Language / Category")
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/bar_chart.png")
    plt.close()

    # 2. Pie Chart for Languages
    pd_lang.loc[pd_lang["percentage"] < 1.0, "language"] = "Other < 1.0%"
    pie_lang = (
        pd_lang.groupby("language", as_index=False)
        .sum()
        .sort_values("percentage", ascending=False)
    )
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap("Set3")
    colors = cmap(range(len(pie_lang)))
    plt.pie(
        pie_lang["percentage"],
        labels=pie_lang["language"],
        autopct="%1.1f%%",
        startangle=140,
        colors=colors,
    )
    plt.title("Tokens by Language (Grouped) - OpenEuroLLM Flagship Mix")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/pie_chart.png")
    plt.close()

    # --- Source Type Analysis ---
    agg_source = (
        df.group_by("source_type")
        .agg(pl.col("budget").sum().alias("total_tokens"))
        .sort("total_tokens", descending=True)
    )

    agg_source = agg_source.with_columns(
        (pl.col("total_tokens") / total_budget * 100).alias("percentage")
    )

    plt.figure(figsize=(10, 8))
    plt.pie(
        agg_source["percentage"],
        labels=agg_source["source_type"],
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Data Sources - OpenEuroLLM Flagship Mix")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/source_types_pie.png")
    plt.close()

    # --- Swedish Breakdown ---
    swe_df = (
        df.filter(pl.col("language") == "swe_Latn")
        .group_by("dataset")
        .agg(pl.col("budget").sum().alias("total_tokens"))
        .sort("total_tokens", descending=True)
    )

    swe_total = swe_df["total_tokens"].sum()
    swe_df = swe_df.with_columns(
        (pl.col("total_tokens") / swe_total * 100).alias("percentage")
    )

    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=swe_df.to_pandas(),
        x="percentage",
        y="dataset",
        hue="dataset",
        palette="magma",
        legend=False,
    )
    plt.title("Sources of Swedish (swe_Latn) Tokens")
    plt.xlabel("Percentage of Total Swedish Tokens (%)")
    plt.ylabel("Dataset")
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/swedish_breakdown.png")
    plt.close()


if __name__ == "__main__":
    analyze_and_visualize("counts.csv", "assets")
