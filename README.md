# CD_Circuit-repro
个人学习项目 - 复现 CD-T 论文以深入理解 Transformer 电路发现机制

本项目是对 [CD-T: Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition](https://arxiv.org/abs/2407.00886) 的学习性复现，仅用于个人学习和研究目的。

## 项目说明
本项目复现了基于 Contextual Decomposition (CD) 的自动化电路发现方法，主要包括：

*   **CD-T 算法**: 利用上下文分解有效地识别模型中的重要组件（电路）。
*   **电路发现**: 自动化地发现 Transformer 模型中特定任务的计算子图。

主要用于学习以下内容：
*   Transformer 内部机制的可解释性
*   Contextual Decomposition (CD) 的原理与实现
*   自动化电路发现 (Automated Circuit Discovery) 的流程
*   如何高效地在大模型上应用相关算法

## 项目结构
```
CD_Circuit-repro/
├── src/              # 核心代码 (待实现/开发中)
├── docs/             # 文档与论文
│   ├── Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition.pdf  # 原始论文
│   ├── HOW_TO_RUN.md
│   └── How_TO_REBUILD.md
├── scripts/          # 运行脚本
├── old/              # 原始参考代码
└── README.md
```

## 学习资源
*   **原始论文**: 见 `docs/` 目录下的 PDF 文件。

## 致谢
本项目基于以下工作进行学习复现：

**原始论文与代码**：
*   **论文**: CD-T: Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition
*   **官方仓库**: [https://github.com/adelaidehsu/CD_Circuit](https://github.com/adelaidehsu/CD_Circuit)

## 引用
如果您的工作参考了 CD-T，请引用原始论文：

```bibtex
@misc{hsu2024efficientautomatedcircuitdiscovery,
      title={Efficient Automated Circuit Discovery in Transformers using Contextual Decomposition}, 
      author={Aliyah R. Hsu and Georgia Zhou and Yeshwanth Cherapanamjeri and Yaxuan Huang and Anobel Y. Odisho and Peter R. Carroll and Bin Yu},
      year={2024},
      eprint={2407.00886},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.00886}, 
}
```

## 声明
*   本项目仅用于个人学习目的
*   所有核心思想和方法归属于原作者
*   如有任何版权问题，请联系删除

## License
本项目遵循原始 CD_Circuit 项目的开源协议。
