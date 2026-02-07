"""
通过pathlib+for循环实现批量抓取与处理，尝试将结果导出为csv
"""
import csv
from pathlib import Path
from pipeline_6Lines import main

def batch():
    list_dict = []
    for path in Path("../data/RAW").rglob("*.png"):
        diameter_eoz = main(path)
        list_dict.append({"ID" : path.stem , "EOZ" : diameter_eoz})
    return list_dict

def save_csv(result, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ID", "EOZ"])
        writer.writeheader()
        writer.writerows(result)

if __name__ == "__main__":
    final = batch()
    print(final)
    save_csv(final, "output-6lines"
                    ".csv")