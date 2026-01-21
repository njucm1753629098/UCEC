import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np
from collections import defaultdict

def analyze_pathway_scores(input_path, output_dir):
    """
    分析路径得分与通路的关系，并生成可视化图表
    
    参数:
    input_path (str): 输入文件或目录路径
    output_dir (str): 输出目录
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")
    
    # 收集所有CSV文件
    csv_files = []
    if os.path.isfile(input_path) and input_path.endswith('.csv'):
        csv_files.append(input_path)
    elif os.path.isdir(input_path):
        for root, dirs, files in os.walk(input_path):
            for file in files:
                if file.endswith('_all_predictions.csv'):
                    csv_files.append(os.path.join(root, file))
    
    if not csv_files:
        print(f"No prediction CSV files found in: {input_path}")
        return
    
    print(f"Found {len(csv_files)} CSV files for analysis")
    
    # 读取并合并所有数据
    all_data = pd.DataFrame()
    for file in csv_files:
        df = pd.read_csv(file)
        print(f"Reading file: {file}")
        print(df.head())  # 打印文件的前几行
        herb_name = os.path.basename(file).split('_')[0]
        df['herb'] = herb_name  # 添加草药名称列
        all_data = pd.concat([all_data, df], ignore_index=True)
    
    # 过滤无效数据
    valid_data = all_data[(all_data['pathway'] != "N/A") & (all_data['path_score'] > 0)]
    
    if valid_data.empty:
        print("No valid pathway data found in the CSV files")
        return
    
    print(f"Total valid pathway records: {len(valid_data)}")
    
    # 设置绘图风格
    sns.set(style="whitegrid", palette="muted", font_scale=1.2)
    plt.figure(figsize=(14, 10))
    
    # 绘制通路得分分布箱线图
    plt.subplot(2, 1, 1)
    pathway_counts = valid_data['pathway'].value_counts()
    top_pathways = pathway_counts.head(15).index.tolist()
    top_data = valid_data[valid_data['pathway'].isin(top_pathways)]
    
    # 按通路出现频率排序
    pathway_order = top_data.groupby('pathway')['path_score'].median().sort_values(ascending=False).index
    ax = sns.boxplot(
        x='pathway', 
        y='path_score', 
        data=top_data,
        order=pathway_order,
        showfliers=False
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Path Score Distribution by Pathway (Top 15 Pathways)', fontsize=16)
    plt.xlabel('Pathway', fontsize=14)
    plt.ylabel('Path Score', fontsize=14)
    plt.tight_layout()
    
    # 绘制通路得分分布小提琴图
    plt.subplot(2, 1, 2)
    sns.violinplot(
        x='pathway', 
        y='path_score', 
        data=top_data,
        order=pathway_order,
        inner="quartile",
        cut=0
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Path Score Density by Pathway (Top 15 Pathways)', fontsize=16)
    plt.xlabel('Pathway', fontsize=14)
    plt.ylabel('Path Score', fontsize=14)
    plt.tight_layout()
    
    # 保存组合图
    output_path = os.path.join(output_dir, 'pathway_score_analysis.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=300)
    print(f"Pathway analysis visualization saved to: {output_path}")
    
    # 分析通路得分与草药的关系
    plt.figure(figsize=(14, 8))
    herb_pathway_scores = valid_data.groupby(['herb', 'pathway'])['path_score'].mean().reset_index()
    
    # 选择得分最高的10个通路
    top_pathways_by_score = herb_pathway_scores.groupby('pathway')['path_score'].mean().nlargest(10).index
    top_herb_pathway = herb_pathway_scores[herb_pathway_scores['pathway'].isin(top_pathways_by_score)]
    
    # 创建热力图
    pivot_table = top_herb_pathway.pivot_table(
        index='herb', 
        columns='pathway', 
        values='path_score', 
        fill_value=0
    )
    
    sns.heatmap(
        pivot_table, 
        annot=True, 
        fmt=".3f", 
        cmap="YlGnBu", 
        linewidths=.5,
        cbar_kws={'label': 'Average Path Score'}
    )
    plt.title('Average Path Score by Herb and Pathway (Top 10 Pathways)', fontsize=16)
    plt.xlabel('Pathway', fontsize=14)
    plt.ylabel('Herb', fontsize=14)
    
    # 保存热力图
    heatmap_path = os.path.join(output_dir, 'herb_pathway_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    print(f"Herb-pathway heatmap saved to: {heatmap_path}")
    
    # 分析通路得分稳定性
    pathway_stability = valid_data.groupby('pathway')['path_score'].agg(['mean', 'std', 'count'])
    pathway_stability['cv'] = pathway_stability['std'] / pathway_stability['mean']  # 变异系数
    pathway_stability = pathway_stability.sort_values(by='cv')
    
    # 保存稳定性分析结果
    stability_path = os.path.join(output_dir, 'pathway_score_stability.csv')
    pathway_stability.to_csv(stability_path)
    print(f"Pathway score stability analysis saved to: {stability_path}")
    
    # 绘制得分稳定性图表
    plt.figure(figsize=(14, 8))
    stable_pathways = pathway_stability[pathway_stability['count'] > 5].nsmallest(15, 'cv')
    
    plt.bar(
        stable_pathways.index, 
        stable_pathways['cv'], 
        color=sns.color_palette("viridis", len(stable_pathways))
    )
    plt.xticks(rotation=45, ha='right')
    plt.title('Path Score Stability (Coefficient of Variation)', fontsize=16)
    plt.xlabel('Pathway', fontsize=14)
    plt.ylabel('Coefficient of Variation (CV)', fontsize=14)
    plt.tight_layout()
    
    # 保存稳定性图表
    stability_plot_path = os.path.join(output_dir, 'pathway_score_stability.png')
    plt.savefig(stability_plot_path, bbox_inches='tight', dpi=300)
    print(f"Pathway stability visualization saved to: {stability_plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Analyze pathway scores from prediction results')
    parser.add_argument('--input', type=str, required=True, 
                       help='Input CSV file or directory containing prediction CSV files')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for analysis results. If not specified, use the same as input.')
    
    args = parser.parse_args()
    
    # 如果没有指定输出路径，则使用输入路径的目录
    if args.output is None:
        args.output = os.path.dirname(args.input)
    
    print("=" * 60)
    print("Pathway Score Analysis")
    print("=" * 60)
    print(f"Input path: {args.input}")
    print(f"Output directory: {args.output}")
    
    analyze_pathway_scores(args.input, args.output)
    
    print("\n" + "=" * 60)
    print("Analysis completed!")
    print("=" * 60)