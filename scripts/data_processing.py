import pandas as pd
import numpy as np
from pathlib import Path
from propy import AAComposition, Autocorrelation, CTD, QuasiSequenceOrder, PseudoAAC
from PyBioMed.PyProtein.PyProtein import PyProtein
import multiprocessing


class BasicDes:
    peptide_information = {
        # '氨基酸符号': np.array([分子量, 电荷, 极性, 芳香性, 疏水性, 等电点])
        'A': [89.1, 0, -1, 0, 0.25, 1], 'R': [174.2, 1, 1, 0, -1.8, 6.13],
        'N': [132.1, 0, 1, 0, -0.64, 2.95], 'D': [133.1, -1, 1, 0, -0.72, 2.78],
        'C': [121.2, 0, 1, 0, 0.04, 2.43], 'Q': [146.2, 0, 1, 0, -0.69, 3.95],
        'E': [147.1, -1, 1, 0, -0.62, 3.78], 'G': [75.1, 0, -1, 0, 0.16, 0],
        'H': [155.2, 1, 1, 0, -0.4, 4.66], 'I': [131.2, 0, -1, 0, 0.73, 4],
        'L': [131.2, 0, -1, 0, 0.53, 4], 'K': [146.2, 1, 1, 0, -1.1, 4.77],
        'M': [149.2, 0, -1, 0, 0.26, 4.43], 'F': [165.2, 0, -1, 1, 0.61, 5.89],
        'P': [115.1, 0, -1, 0, -0.07, 2.72], 'S': [105.1, 0, 1, 0, -0.26, 1.6],
        'T': [119.1, 0, 1, 0, -0.18, 2.6], 'W': [204.2, 0, -1, 1, 0.37, 8.08],
        'Y': [181.2, 0, 1, 1, 0.02, 6.47], 'V': [117.2, 0, -1, 0, 0.54, 3]
    }
    
    @staticmethod
    def calc_descriptor(peptide: str):
        features = np.array([BasicDes.peptide_information[aa] for aa in peptide])
        
        positive_charge = np.sum(features[:, 1] == 1)
        negative_charge = np.sum(features[:, 1] == -1)
        polar_number = np.sum(features[:, 2] == 1)
        unpolar_number = np.sum(features[:, 2] == -1)
        ph_number = np.sum(features[:, 3])
        weight = np.sum(features[:, 0]) - (len(peptide) - 1) * 18  # 减去水分子质量
        charge_of_all = np.sum(features[:, 1])
        hydrophobicity = np.mean(features[:, 4])
        van_der_Waals_volume = np.mean(features[:, 5])
        
        pep_discriptor = {
            'Mw': weight,
            'charge of all': charge_of_all,
            'positive_charge': positive_charge,
            'negative_charge': negative_charge,
            'polar_number': polar_number,
            'unpolar_number': unpolar_number,
            'ph_number': ph_number,
            'hydrophobicity': hydrophobicity,
            'vdW_volume': van_der_Waals_volume,
        }
        return pep_discriptor




class DataGenerator():
    def __init__(self, pos_data_path, neg_data_path, processed_data_dir, target_bacterium, length_min, length_max, num_workers=4):
        """初始化 GenerateSample 类。

        Args:
            pos_data_path (str): 正样本数据文件路径。
            neg_data_path (str): 负样本数据文件路径。
            processed_data_path (str): 输出数据存储路径。
            target_bacterium (str): 目标细菌名称，仅用于筛选相关样本。
            length_min (int): 过滤的最小序列长度。
            length_max (int): 过滤的最大序列长度。
            mode (str): 数据生成模式，'all' 表示包含所有样本，'pos' 表示仅处理正样本。

        Raises:
            ValueError: 如果 `mode` 不是 'all' 或 'pos'，则抛出异常。
        """
        self.pos_data_path = pos_data_path
        self.neg_data_path = neg_data_path
        self.processed_data_dir = processed_data_dir
        self.target_bacterium = target_bacterium
        self.length_min = length_min
        self.length_max = length_max
        self.num_workers = num_workers
    
    def __call__(self, *args, **kwargs):
        pos_data = pd.read_csv(self.pos_data_path, index_col=0)
        neg_data = pd.read_csv(self.neg_data_path)
        
        filtered_pos_data = self.pos_data_filter(pos_data, self.target_bacterium, self.length_min, self.length_max)
        filtered_neg_data = self.neg_data_filter(neg_data, self.length_min, self.length_max)
        
        generated_pos_data = self.gen_pos_data(filtered_pos_data)
        generated_neg_data = self.gen_neg_data(filtered_neg_data)

        print(f"Positive samples: {len(generated_pos_data)}")
        print(f"Negative samples: {len(generated_neg_data)}")

        all_data = pd.concat([generated_pos_data, generated_neg_data], ignore_index=True)
        print(f"Total samples: {len(all_data)}")
        
        all_sequences = all_data["sequence"].tolist()
        
        all_descriptors = self.gen_peptide_descriptor_parallel(all_sequences, self.num_workers)
        all_feature_df = pd.concat([all_data.iloc[:, 0], pd.DataFrame(all_descriptors), all_data.iloc[:, 1:]], axis=1)
        pos_feature_df = all_feature_df[all_feature_df['type'] == 1]

        all_feature_df.to_csv(self.processed_data_dir / 'all_feature.csv', index=False)
        print(f"All features saved to {self.processed_data_dir / 'all_feature.csv'}")
        pos_feature_df.to_csv(self.processed_data_dir / 'pos_feature.csv', index=False)
        print(f"Positive features saved to {self.processed_data_dir / 'pos_feature.csv'}")
        return all_feature_df, pos_feature_df

        
    def pos_data_filter(self, data, target_bacterium, length_min, length_max, has_cterminal_amidation=True):
        data = data[data['bacterium'] == target_bacterium]
        data = data[(data['sequence'].str.len() >= length_min) & (data['sequence'].str.len() <= length_max)]
        if has_cterminal_amidation:
            data = data[data['has_cterminal_amidation'] == True]
        data = data.drop_duplicates()
        return data

    def neg_data_filter(self, data, length_min, length_max):
        data = data[(data['Sequence'].str.len() >= length_min) & (data['Sequence'].str.len() <= length_max)]
        data = data[~data["Sequence"].str.contains("[BXZOU]", regex=True)]
        data = data.drop_duplicates()
        return data

    @staticmethod
    def geometric_mean(series):
        return np.mean(np.power(10, series))

    def gen_pos_data(self, data): 
        # 按照 sequence 分组并计算几何平均 MIC
        formatted_data = (
            data.groupby("sequence")["value"]
            .apply(self.geometric_mean)
            .reset_index()
            .rename(columns={"value": "MIC"})
            )
        
        # 添加类型列（正样本标记为 1）
        formatted_data["type"] = 1
        return formatted_data
    
    def gen_neg_data(self, data):
        formatted_data = data[["Sequence"]].drop_duplicates().rename(columns={"Sequence": "sequence"}).reset_index(drop=True)
        formatted_data["MIC"] = 8196
        formatted_data["type"] = 0
        return formatted_data
    
    @staticmethod
    def gen_peptide_descriptor(peptide: str):
        AAC = AAComposition.CalculateAAComposition(peptide)  # 氨基酸组成
        DIP = AAComposition.CalculateDipeptideComposition(peptide)  # 二肽组成
        MBA = Autocorrelation.CalculateNormalizedMoreauBrotoAutoTotal(peptide)  # Moreau-Broto 自相关
        CCTD = CTD.CalculateCTD(peptide)  # CTD 组合描述符
        QSO = QuasiSequenceOrder.GetSequenceOrderCouplingNumberTotal(peptide, maxlag=5)  # 序列顺序耦合数
        PAAC = PseudoAAC._GetPseudoAAC(peptide, lamda=5)  # 伪氨基酸组成（PAAC）
        APAAC = PseudoAAC.GetAPseudoAAC(peptide, lamda=5)  # 修正伪氨基酸组成（APAAC）
        Basic = BasicDes.calc_descriptor(peptide)  # 基本描述符
        descriptors = {**AAC, **DIP, **MBA, **CCTD, **QSO, **PAAC, **APAAC, **Basic}
        return descriptors
    
    @staticmethod
    def _process_wrapper(x):
        return x[0], DataGenerator.gen_peptide_descriptor(x[1])

    @staticmethod
    def gen_peptide_descriptor_parallel(peptides: list, num_workers):    
        # 为每个肽分配一个索引，用于保持顺序
        indexed_peptides = list(enumerate(peptides))
        ordered_results = [None] * len(peptides)  # 预分配列表，用于按原始顺序存储结果
        with multiprocessing.Pool(num_workers) as pool:
            for i, (original_idx, descriptor) in enumerate(
                pool.imap_unordered(
                    DataGenerator._process_wrapper, 
                    indexed_peptides
                ), 
                1
            ):
                # 将结果存入原始索引位置
                ordered_results[original_idx] = descriptor

                if i % 1000 == 0:
                    print(f"已处理 {i}/{len(peptides)} 条序列")

        print(f"所有序列处理完成！共 {len(ordered_results)} 条")
        return ordered_results
    

if __name__ == '__main__':
    pos_data_path = Path('data/raw/pos.csv')
    neg_data_path = Path('data/raw/neg.csv')
    processed_data_dir = Path('data/processed')
    data_generator = DataGenerator(
        pos_data_path=pos_data_path,
        neg_data_path=neg_data_path,
        processed_data_dir=processed_data_dir,
        target_bacterium='S. aureus',
        length_min=6,
        length_max=50,
        num_workers=4
    )
    all_feature_df, pos_feature_df = data_generator()
