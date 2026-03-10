import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set up plot style
sns.set_theme(style="ticks", font_scale=1.1)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

class NeighborhoodAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.cleaned_df = None

    def load_data(self):
        print("Loading data...")
        self.df = pd.read_csv(self.file_path)

    def clean_data(self):
        print("\nCleaning data...")
        self.cleaned_df = self.df.copy()

        # Calculate density metrics
        self.cleaned_df['airbnb_density'] = (self.cleaned_df['airbnb_count'] / 
                                           self.cleaned_df['population']) * 1000
        self.cleaned_df['population_density'] = self.cleaned_df['population'] / 9

        # Remove rows with missing values
        self.cleaned_df = self.cleaned_df.dropna(subset=['median_income', 'population'])

        # Calculate non-entire home ratio for stacked charts
        if 'entire_home_ratio' in self.cleaned_df.columns:
            self.cleaned_df['other_room_ratio'] = 1 - self.cleaned_df['entire_home_ratio']

    def analyze_income_vs_airbnb_density(self):
        print("\nGenerating: Income vs Density joint plot...")

        g = sns.jointplot(
            data=self.cleaned_df,
            x='median_income',
            y='airbnb_density',
            hue='city',
            palette='Set2',
            alpha=0.75,
            s=80,
            edgecolor='w',
            linewidth=0.5,
            height=8,
            ratio=5
        )

        # Add global trend line
        sns.regplot(
            data=self.cleaned_df,
            x='median_income',
            y='airbnb_density',
            scatter=False,
            ax=g.ax_joint,
            color='coral',
            line_kws={'linewidth': 2, 'linestyle': '--'}
        )

        g.set_axis_labels('Median Income ($)', 'Airbnb Density (per 1000 people)')
        g.fig.suptitle('Income vs Airbnb Density with Marginal Distributions', y=1.03, weight='bold')
        g.savefig('1_income_vs_airbnb_density_joint.png', dpi=300, bbox_inches='tight')

    def analyze_population_density_vs_airbnb_density(self):
        print("Generating: Population Density vs Airbnb Density plot...")

        g = sns.jointplot(
            data=self.cleaned_df,
            x='population_density',
            y='airbnb_density',
            hue='city',
            palette='Set2',
            alpha=0.75,
            s=80,
            edgecolor='w',
            height=8
        )

        # Add global regression line
        sns.regplot(
            data=self.cleaned_df,
            x='population_density',
            y='airbnb_density',
            scatter=False,
            ax=g.ax_joint,
            color='crimson',
            line_kws={'linewidth': 2}
        )

        g.set_axis_labels('Population Density (people/sq mile)', 'Airbnb Density (per 1000 people)')
        g.fig.suptitle('Urban Density vs Airbnb Penetration', y=1.03, weight='bold')
        g.savefig('2_population_density_vs_airbnb_density.png', dpi=300, bbox_inches='tight')

    def analyze_structure_comparison(self):
        print("Generating: Property structure comparison plots...")

        # 100% stacked bar chart
        city_stats = self.cleaned_df.groupby('city')[['entire_home_ratio', 'other_room_ratio']].mean().reset_index()
        cities = city_stats['city']
        entire = city_stats['entire_home_ratio'] * 100
        other = city_stats['other_room_ratio'] * 100

        fig, ax = plt.subplots(figsize=(10, 6))

        # Create stacked bars
        ax.bar(cities, entire, label='Entire Home / Apt', color='#4c72b0', edgecolor='white', width=0.5)
        ax.bar(cities, other, bottom=entire, label='Private / Shared Room', color='#dd8452', edgecolor='white', width=0.5)

        ax.set_title('Property Type Structure by City (100% Stacked)', pad=20, weight='bold')
        ax.set_ylabel('Percentage (%)')

        # Add percentage labels
        for i, city in enumerate(cities):
            ax.text(i, entire.iloc[i] / 2, f"{entire.iloc[i]:.1f}%", ha='center', va='center', color='white', weight='bold')
            ax.text(i, entire.iloc[i] + other.iloc[i] / 2, f"{other.iloc[i]:.1f}%", ha='center', va='center', color='white', weight='bold')

        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1), frameon=False)
        sns.despine(left=True, bottom=True)
        plt.savefig('3_property_structure_stacked_bar.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Violin plot with scatter overlay
        plt.figure(figsize=(10, 6))
        sns.violinplot(
            data=self.cleaned_df,
            x='city',
            y='entire_home_ratio',
            palette='Pastel1',
            inner='quartile',
            linewidth=1.2
        )
        
        # Add scatter plot overlay
        sns.stripplot(
            data=self.cleaned_df,
            x='city',
            y='entire_home_ratio',
            color='black',
            alpha=0.25,
            size=5,
            jitter=True
        )
        
        plt.title('Distribution of Entire Home Ratios Across Zip Codes', pad=15, weight='bold')
        plt.ylabel('Entire Home Ratio')
        plt.xlabel('')
        sns.despine()
        plt.savefig('4_entire_home_ratio_violin.png', dpi=300, bbox_inches='tight')
        plt.close()

    def analyze_additional_metrics(self):
        print("Generating: Income-Price bubble plot and heatmap...")

        # Income vs Price bubble plot (with outlier filtering)
        plt.figure(figsize=(12, 7))
        # Filter out extreme price outliers to prevent chart compression
        df_filtered = self.cleaned_df[self.cleaned_df['avg_airbnb_price'] < 2000]

        scatter = sns.scatterplot(
            data=df_filtered,
            x='median_income',
            y='avg_airbnb_price',
            hue='city',
            size='airbnb_count',  # Bubble size represents total Airbnb count in zip code
            sizes=(30, 800),
            alpha=0.6,
            palette='Set1',
            edgecolor="w",
            linewidth=1
        )
        
        plt.title('Income vs Price (Bubble Size = Total Airbnb Count)', pad=15, weight='bold')
        plt.xlabel('Median Income ($)')
        plt.ylabel('Average Airbnb Price ($) - Outliers Removed')

        # Optimize legend position
        h, l = scatter.get_legend_handles_labels()
        plt.legend(h, l, bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
        sns.despine()
        plt.savefig('5_income_vs_price_bubble.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Lower triangle correlation heatmap
        corr = self.cleaned_df[[
            'airbnb_density', 'median_income', 'population_density',
            'entire_home_ratio', 'avg_airbnb_price', 'avg_review_score'
        ]].corr()

        # Create mask to show only lower triangle
        mask = np.triu(np.ones_like(corr, dtype=bool))

        plt.figure(figsize=(10, 8))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)

        sns.heatmap(
            corr,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .7},
            annot=True,
            fmt='.2f',
            annot_kws={"size": 11, "weight": "bold"}
        )
        
        plt.title('Neighborhood Characteristics Correlation Matrix', pad=20, weight='bold')
        plt.tight_layout()
        plt.savefig('6_correlation_heatmap_lower.png', dpi=300)
        plt.close()

    def run_analysis(self):
        self.load_data()
        self.clean_data()
        self.analyze_income_vs_airbnb_density()
        self.analyze_population_density_vs_airbnb_density()
        self.analyze_structure_comparison()
        self.analyze_additional_metrics()

if __name__ == "__main__":
    analyzer = NeighborhoodAnalyzer('final.csv')
    analyzer.run_analysis()