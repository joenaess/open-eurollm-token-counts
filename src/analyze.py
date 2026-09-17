import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

def extract_language(part: str) -> str:
    """Extract language from the 'part' column."""
    if part is None:
        return 'Other'
        
    part_str = str(part)
    tags = re.findall(r'[a-z]{3}_[A-Z][a-z]{3}', part_str)
    
    if not tags:
        lower_part = part_str.lower()
        if 'math' in lower_part:
            return 'Math'
        if 'code' in lower_part:
            return 'Code'
        return 'Other'
        
    if len(tags) == 1:
        return tags[0]
        
    # If multiple tags, pick the non-English one if it involves English
    for tag in tags:
        if tag != 'eng_Latn':
            return tag
            
    return tags[0]

def analyze_and_visualize(csv_path: str, assets_dir: str):
    """Load data, analyze languages, and create visualizations."""
    os.makedirs(assets_dir, exist_ok=True)
    df = pl.read_csv(csv_path)
    
    # Process dataset
    df = df.with_columns(
        pl.col("part").fill_null("").map_elements(extract_language, return_dtype=pl.Utf8).alias("language")
    )
    
    # Aggregate tokens by language (using budget)
    # Ensure budget is treated as numeric
    df = df.with_columns(pl.col("budget").cast(pl.Float64))
    
    agg_df = df.group_by("language").agg(
        pl.col("budget").sum().alias("total_tokens")
    ).sort("total_tokens", descending=True)
    
    total_budget = agg_df["total_tokens"].sum()
    
    agg_df = agg_df.with_columns(
        (pl.col("total_tokens") / total_budget * 100).alias("percentage")
    )
    
    print("Aggregation complete. Here are the top languages:")
    print(agg_df.head(10))
    
    # Generate Visualizations
    sns.set_theme(style="whitegrid")
    
    # 1. Bar Chart
    plt.figure(figsize=(12, 10))
    pd_df = agg_df.to_pandas()
    sns.barplot(data=pd_df, x="percentage", y="language", hue="language", palette="viridis")
    plt.title("Tokens by Language (%) - OpenEuroLLM Flagship Mix")
    plt.xlabel("Percentage of Total Tokens (%)")
    plt.ylabel("Language / Category")
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/bar_chart.png")
    plt.close()
    
    # 2. Pie Chart (Group smaller ones into 'Other Languages' for better visibility if too many)
    # Mask anything less than 1.5% as 'Other < 1.5%'
    pd_df.loc[pd_df['percentage'] < 1.5, 'language'] = 'Other < 1.5%'
    pie_df = pd_df.groupby('language', as_index=False).sum().sort_values('percentage', ascending=False)
    
    plt.figure(figsize=(10, 10))
    cmap = plt.get_cmap('Set3')
    colors = cmap(range(len(pie_df)))
    plt.pie(pie_df['percentage'], labels=pie_df['language'], autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title("Tokens by Language (Grouped) - OpenEuroLLM Flagship Mix")
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"{assets_dir}/pie_chart.png")
    plt.close()
    
    # Save the aggregated data to CSV for easy inspection
    agg_df.write_csv(f"{assets_dir}/aggregated_tokens.csv")

if __name__ == "__main__":
    analyze_and_visualize("counts.csv", "assets")
