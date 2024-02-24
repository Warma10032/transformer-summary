---
layout: all
title: Transformer经典论文阅读后综述
author: 小包子
cover: 
description: Transformer模型是一种革命性的神经网络架构，它在自然语言处理（NLP）和计算机视觉（CV）等多个领域都取得了卓越的成就。本论文旨在介绍Transformer模型的结构、以及在NLP领域和CV领域中的应用。深入探讨Transformer模型在不同领域之间的共同特点和差异。我们首先详细介绍了Transformer模型的核心结构，然后探讨了在NLP任务中的Transformer变种（如BERT和GPT），最后研究了Transformer在计算机视觉中的新兴应用（如Vision Transformer）。通过本文，读者将能够深入了解Transformer模型的工作原理以及它在不同领域中应用的优势和改进。
toc_number: true
comments: true
typora-root-url: Transformer经典论文阅读后综述/
tags:
  - Transformer
  - AI
  - 人工智能
  - 论文
  - gpt
categories:
  - - 人工智能
  - - AI
date: 2024-02-24 22:42:28
---

# Transformer in Deep Learning



## 摘要

Transformer模型是一种革命性的神经网络架构，它在自然语言处理（NLP）和计算机视觉（CV）等多个领域都取得了卓越的成就。本论文旨在介绍Transformer模型的结构、以及在NLP领域和CV领域中的应用。深入探讨Transformer模型在不同领域之间的共同特点和差异。我们首先详细介绍了Transformer模型的核心结构，然后探讨了在NLP任务中的Transformer变种（如BERT和GPT），最后研究了Transformer在计算机视觉中的新兴应用（如Vision Transformer）。通过本文，读者将能够深入了解Transformer模型的工作原理以及它在不同领域中应用的优势和改进。



## 引言

自然语言处理（NLP）和计算机视觉（CV）一直是人工智能领域的两大关键领域。这些领域的发展一直受制于模型的性能和能力，而Transformer模型的出现冲击了RNN和CNN的统治地位。在过去的几年里，Transformer模型已经成为NLP和CV任务的首选模型之一，且具有统一NLP和CV研究方法的势头。

首先，我们将详细介绍Transformer模型的基本结构，包括自注意力机制（self-attention）。这种机制使其能够同时处理输入序列中的各个元素，无论是单词、像素还是其他形式的数据。这个思想的强大之处在于它的通用性，它使Transformer能够在各种领域中取得成功。在NLP领域，Transformer模型已在文本分类、命名实体识别和机器翻译等领域取得了巨大成功。而在计算机视觉领域，Transformer也已经在图像分类和目标检测中崭露头角。

然而，虽然Transformer模型在不同领域之间的应用非常广泛，但它在每个领域中都有自己的独特挑战和特点。例如，在NLP中，BERT模型通过让模型获得双向输入，掌握上下文信息来改善预训练词嵌入，而在CV中，ViT模型通过对图像的分块处理来应对图像数据的巨大复杂性。通过对这些异同之处的深入研究，我们可以更好地理解Transformer模型的多领域适用性，以及如何将其推向新的高度。



## 什么是Transformer

Transformer是一个基于注意力机制的神经网络模型，Transformer模型由编码器和解码器组成，两者都由多个相同的层组成，每一层有两个子层：多头自注意力机制层和全连接前馈神经网络，Transformer模型利用自注意力机制捕捉输入序列中每个元素与其他元素之间的关系，利用多头机制提高模型的性能。

![Transformer架构](transformer结构.png)

接下来我将自下而上从输入到输出介绍transformer模型。



### Input Embedding and Output Embedding（输入嵌入和输出嵌入）

Word Embedding（词嵌入）是将离散化的符号序列（如单词、字符）转换为连续的向量表示，以便使神经网络可以有效地处理文本数据。它使得模型能够理解文本数据的语义和上下文信息，并为后续的自注意力层和前馈神经网络提供了输入。



### Positional Encoding（位置编码）

由于 Transformer 不包含处理输入序列的循环或卷积操作，它不能自动地理解单词的顺序信息。因此，我们必须注入一些关于相对或绝对位置的信息序列中的标记。为此，我们引入了“位置编码”来将位置信息嵌入到模型中。位置编码是一个与位置、维度相关的矩阵，它会与输入的词嵌入相加。在原论文中，作者采用了正余弦函数来进行位置编码。



### Attention（注意力机制）

注意力函数可以描述为将查询（query）和一组键值对（key-value pairs）映射到输出，其中查询(Q)、键(K)、值(V)和输出都是向量。 输出被计算为值的加权和，其中分配给每个值的权重是由查询(Q)与相应键(v)的兼容性函数计算的。

![注意力函数](注意力函数.png)                

#### Scaled Dot-Product Attention（缩放点积注意力）

自注意力机制（Self-attention）是 Transformer 模型的关键组成部分，它允许模型在一个序列中的不同位置之间建立权重连接，从而在一个步骤内同时考虑所有位置的信息。自注意力机制的计算分为以下几步：

- 对于一个输入序列，首先通过三个线性变换分别得到查询(Q)、键(K)和值(V)的向量表示。这些向量用于计算注意力分数和生成权重。

- 计算注意力分数：为了衡量每个位置与其他位置的关联程度，通过计算查询和键之间的点积，然后进行缩放（通常使用缩放因子，如$\sqrt{d_k}$），得到注意力分数（Attention Scores）。

- 注意力分数的归一化（Normalization）：应用 Softmax 函数将注意力分数转化为概率分布，使得每个位置对其他位置的贡献权重为1，形成归一化的注意力权重。

- 加权求和：将注意力权重应用于值的向量，然后将所有加权的值进行加和，得到最终的输出。这个输出包含了所有位置的信息，但每个位置的贡献由其与其他位置的关联程度决定。

自注意力机制允许模型动态地分配不同位置的权重，从而在不同任务中灵活捕获上下文信息。这是 Transformer 模型在各种自然语言处理任务中表现出色的关键之一。

![自注意力机制](自注意力机制.png)

#### Multi-Head Attention（多头注意力机制）

Transformer 中的多头注意力机制（Multi-Head Attention）是自注意力机制的扩展，允许模型在不同表示子空间中学习自注意力。这个机制增加了模型的表示能力，使其能够同时关注输入序列中的不同信息，从而更好地捕捉序列中的复杂关系。多头注意力机制通过将多个并行的自注意力机制组合在一起，每个自注意力机制都称为一个“头”。每个头有自己的一组查询、键和值参数，这些参数是通过学习而得的。将所有注意力头的输出连接在一起，并通过一个线性变换来产生最终的多头注意力输出。其优点在于提高了模型的学习能力：不同的头可以学习捕获不同的关系，从而提高了模型的表示能力。例如，一些头可以关注语法关系，而其他头可以关注语义关系。提高了训练效率：多头注意力可以并行计算，因为每个头都是独立的，这提高了模型的训练和推理效率。提高了模型稳定性：多头注意力有助于减轻注意力机制中的一些不稳定性，因为多个头的组合可以平滑化注意力权重的分布。

![多头注意力机制](多头自注意力机制.png)

#### Add & Norm（残差连接&层归一化）

Transformer在每个子层（自注意力层和前馈神经网络层）的输入和输出之间，引入了残差连接和层归一化，残差连接可以通过跳过该层来传递一定量的信息，使梯度直接流过某层；层归一化把数据转化成均值为 0 方差为1的数据，保证数据特征分布的稳定性。这些方法可以帮助缓解训练过程中的梯度消失问题。

#### Feed-Forward Neural Network（前馈神经网络）

Transformer的每个编码器和解码器层都包括一个前馈神经网络。它将自注意力层的输出映射到一个更高维度的空间，然后再映射回原始维度。这个过程包括两个线性变换和一个非线性激活函数，通常是ReLU。其作用是通过线性变换，先将数据映射到高纬度的空间再映射到低纬度的空间，提取了更深层次的特征。并且加入一定非线性变换，提高模型学习能力。

#### Masked Multi-Head Attention（带掩码的多头注意力机制）

Masked Multi-Head Attention是一种用于处理可变长度序列的重要技术。它的主要作用是在自注意力计算中，确保模型只关注当前位置之前的信息，而不会泄露未来位置的信息。在处理序列数据时，尤其是在解码器中，我们会用到掩码机制来确保生成的每个位置的信息只取决于当前位置及其之前的信息。这是因为在生成输出序列时，未来的信息是不可见的。具体来说，掩码机制会创建一个掩码矩阵，该矩阵与注意力分数相乘，将未来位置的分数置为负无穷大（或经过 softmax 后为零），从而在注意力计算中消除了未来位置的影响。这使得模型只能关注于当前位置以及之前的信息，而不关注当前位置之后的信息。



### Transformer的优势

- 与RNN和CNN相比，Transformer具有更好的并行性：Transformer中的自注意力机制可以并行计算，而RNN和CNN中的循环结构和卷积结构需要依次计算，难以并行化。例如RNN需要从循环结构中获取序列的时间依赖关系，而CNN需要从小区域到大区域依次扩大感受野。

- 更好的长距离依赖建模能力：Transformer中的自注意力机制可以捕捉输入序列中任意两个位置之间的关系，而RNN只能捕捉相邻位置之间的关系，CNN只能捕捉局部区域内的关系。

- 更好的全局特征获取能力：Transformer中的自注意力机制可以对输入序列中每个位置进行注意力计算，从而获取全局上下文信息，而RNN和CNN只能获取局部上下文信息。鉴于Transformer相较于之前的模型有这么多优点，它很快被用在了NLP，CV等领域上。



## Transformer in NLP

Transformer被提出时，是被用在机器翻译上的。它帮助改善了RNN在并行性和长文本处理上的缺陷，使得机器翻译模型的准确率提高了几个百分点。当人们意识到transformer和attention在NLP上具有很多优势，便想把transformer扩展应用文本分类、语义分析、语言生成等领域。他们将transformer 与预训练结合，训练出一个模型，再通过微调去适应不同的任务。这样可以减少模型在特定任务上的训练时间和数据量，并且降低过拟合的发生模型更加准确和可靠。于是在NLP中，预训练transformer开始大行其道。



### Bert: :Bidirectional Encoder Representation from Transformers

![Bert架构](bert架构.png)

Bert 是一个预训练的语言表征模型。其主要用到了transformer 的Encoder 模块，在预训练时采用了Masked Language Model（MLM）和Next Sentence Prediction（NSP）方法，强调双向获取信息，充分利用上下文信息，提高模型的理解能力。下面我将就Bert的一些改进进行介绍。

#### Encoder only

Bert作为一个特征提取器，其主要目的是进行自监督的预训练学习，学习文本的深层语义信息。因此它仅保留了transformer中的Encoder部分，并且借助Encoder不掩盖后面位置的信息的特点，Bert实现了让模型学习双向的文本信息，结合上下文进行预测。

#### Masked Language Model

MLM是 BERT 模型的预训练任务之一。在 MLM 任务中，输入文本的一部分词汇会被随机遮盖，模型的任务是根据上下文来预测这些被遮盖的词汇。首先，它随机地将输入文本中的15%的词替换为一个特殊的[MASK]标记，表示这些词被遮盖了。然后，它使用一个基于Transformer的编码器来处理这个遮盖后的文本，得到每个位置的隐藏状态向量。最后，将每个位置的隐藏状态向量输入一个softmax来得到词汇表中所有词的概率分布，进而预测被覆盖的词。MLM任务使得Bert不是像之前的语言模型那样只能从左到右或者从右到左地处理文本。这对于很多自然语言处理的下游任务，如机器阅读理解，自然语言推理，问答系统等，都有很大的帮助。

#### Next Sentence Prediction

NSP也是BERT 模型的预训练任务之一。在NSP任务中，模型会输入两个句子，模型需要判断这两个句子是否具有上下文关联关系。首先，它从文本语料库中随机选择两个句子作为输入，其中第一个句子称为A，第二个句子称为B。然后，它在两个句子之间加入一个特殊的分隔符[SEP]，并在句子A的开头加入一个特殊的token [CLS]，作为整个输入序列的第一个词。接着，它使用一个基于Transformer的编码器来处理这个输入序列，得到每个位置的隐藏状态向量。最后，它使用一个二分类器来预测句子B是否在原始文档中是句子A的下一句。在训练期间，输入的句子对50％ 在原始文档中是前后关系，另外 50％ 是从语料库中随机组成的无关句子对。

#### Improve of input embeddings

Bert的input embeddings相较于Transformer进行了改进，它将Position Embeddings不再由公式进行转换，而是让模型去学习如何对位置进行编码。并且Bert加入了Segment Embeddings，用来表示两个不同的句子，以匹配它的预训练任务。Bert还开创性的在input embeddings中加入了一些特殊token。如[CLS]，[SEP]，[CLS]经过自注意力机制后的输出就是模型对一整句话的注意力，即模型对句子的特征提取。因为注意力机制的特殊性，把它放在句子中的任意位置效果都相同。而[SEP]则是用来分隔不同的句子的。这些特殊字符在模型中有特殊的含义，后续的ViT模型也借鉴了这种用特殊字符的方法来提高模型的理解能力。

![The input embeddings in Bert](输入嵌入.png)

#### Fine-tuning（微调）

前面提到，Bert本质是一个预训练的特征提取器。它将输入的句子，通过transformer的Encoder进行编码，将句子的特征提取出来。为了将这些句子特征运用到各种自然语言处理任务上，还需要对模型进行微调。比如选择特定的目标函数，将句子的哪些部分的特征作为下游任务的输入等等。在【Bert框架图】中，就具体展示了Bert如何在SQuAD数据集上进行问答任务。

#### The contribution of Bert

Bert模型的贡献在于它引入了预训练和微调体系和开创了双向语义理解的预训练任务。Bert模型只需要在大量无标注的数据上进行无监督训练，然后进行模型迁移和有监督微调，便可应用在不同的任务上。这大大减少了对特定任务的模型训练成本和大量有标注数据的需求。而双向语义理解的预训练任务则很好的提高了模型对上下文结合的理解能力。



### GPT: Generative Pre-trained Transformer

GPT是一种基于Transformer的生成式预训练语言模型，它可以根据给定的文本生成连贯和自然的文本，并且可以用于多种自然语言处理的任务，如机器翻译，文本摘要，问答系统等。GPT的核心思想是利用大量的无标注文本数据进行无监督的预训练，学习文本的通用表示，然后在特定的任务上进行有监督的微调，调整模型参数，提高模型性能。GPT有多个版本，如GPT-1，GPT-2，GPT-3等，每个版本都增加了模型的参数数量和训练数据的规模，从而提高了模型的生成能力和泛化能力。下面我将就GPT-1到GPT-3的一些研究论文，介绍一下GPT的特点。

#### Decoder only

![](gpt架构.png)

GPT只使用了Transformer的Decoder部分，而没有使用Encoder部分。这是因为GPT的目标是做语言建模，即根据上文预测下一个单词。而语言建模只需要关注之前生成的文本，而不需要关注整个输入序列。因此，GPT只需要使用掩码自注意力层来实现这个功能，而不需要使用编码器-解码器注意力层。另外，GPT还去掉了Decoder中原本用来接收Encoder输出的多头自注意力层，因为GPT没有Encoder部分。GPT和Transformer Decoder的结构对比如图。

#### GPT-2 can be a zero-shot learning language model

在模型大小和训练参数量不断扩大的情况下，GPT-2做到了zero-shot learning（零样本学习）。它可以在没有任何标注数据或者参数调整的情况下，直接执行下游任务。这一点与BERT等模型不同，它们需要在预训练之后进行有监督的微调才能适应特定的任务。当GPT-2要执行一个zero-shot的下游任务时，它只需要根据任务的描述和示例来生成相应的文本。比如，如果要做机器翻译任务，你只需要输入“Translate from English to French: Hello, how are you?”，它就会输出“Bonjour, comment allez-vous?”。如果要做问答任务，你只需要输入“Who is the president of U.S.?”，它就会输出“Joseph Robinette Biden Jr.”。这些输入都可以看作是给模型的提示或者指令（prompt），让模型知道要做什么样的任务，而不需要再去给一些有标注数据让模型进行迁移学习（微调），提高了模型的性能和泛化能力。这是因为GPT-2使用了海量的无标注文本数据进行预训练，学习了通用的语言知识和规律。这些数据包含了各种任务相关的信息，比如语法，逻辑，常识等，让模型可以理解输入的自然语言的语义。GPT-2进而通过自回归的方式，从左到右地预测下一个单词，从而进行语言生成和输出。

#### A particularly large GPT-3 exhibits emergence

在GPT-2之后，OpenAI进一步的增加模型大小和训练参数量，最后在一种量变导致质变的情况下，模型的性能取得了质的飞跃，出现了Grokking（”涌现“）。例如模型的准确率和泛化性大大提高，可以通过少量的示例来完成各种下游任务。出现“涌现”的原因尚不可知，人们猜测是因为大模型具有更强的记忆和推理能力，能够从海量的数据中学习到更多的知识和规律。总之在该论文中，GPT-3在参数量提升到1750亿的情况下，模型的能力刷新了很多当时NLP领域的记录。并且论文指出，GPT模型还未见到增加参数量带来的性能饱和现象，这说明继续增加参数量和原始数据，GPT的性能还能进一步的提升。

#### Where to get so much text data

GPT需要大量的文本数据用来学习，如何获得高质量的文本数据也是一个值得研究的方向。OpenAI在GPT-2中使用了一个国外论坛Reddit的数据，并借助Reddit的点赞机制，筛选优质数据。而GPT-3训练所用的数据量又翻了1125倍，来到了45TB，相当于互联网上所有文本的10%。GPT-3借助Common Crawl从互联网爬取的超大数据集，并将Common Crawl进行筛选过滤。其过滤算法是借助GPT-2已筛选好的优质文本作为正例训练LR分类器，用来筛选数据。再通过去重等方法得到优质数据集。

###  The Advantages of Pre-trained Transformer in NLP

由上文可知Transformer预训练模型可以充分利用大规模的无标注文本数据，通过自监督学习的方式，学习到通用的语言知识和表示，从而提高下游任务的性能和泛化能力。并且Transformer预训练模型具有很好的可扩展性和灵活性，可以根据不同的任务需求，调整模型的结构和参数，实现多种功能和应用。Transformer预训练模型是目前NLP领域最先进和最流行的技术之一，它已经在机器翻译、文本分类、阅读理解、对话生成、文本摘要等多个任务上取得了显著的效果，展现了巨大的潜力和前景。



##  Transformer in CV

在CV领域，transformer也在大放异彩。一开始，人们将transformer中的attention与CNN结合，将attention用于CNN中的特征提取。后来Vision Transformer这篇论文提出，用纯的transformer结构进行预训练，训练出来的模型同样具有很好的效果。

![ViT架构](ViT架构.png)

### How to enter the image into the Transformer

如何将二维的图片转化为一维的序列输入transformer是想要在CV中利用transformer的第一步。这篇论文展示了一种巧妙的方法。即将图片分为一个个patch，将patchs排成一维序列输入transformer便达到了类似NLP中输入一句话的效果。这也是论文表示为什么说AN IMAGE IS WORTH 16X16 WORDS 的原因。至于为什么不将图片的每个像素分割拉直成一维，那是以为这样处理后的序列将会变得很长，增加了模型训练的复杂度。

### Patch and Position Embedding

将图片分割成一个个patch之后势必会损失图片的某些位置信息。不过好在transformer模型本身就提供了一个Position Embedding，这使得Patch Embedding后的向量可以再加上一个Position Embedding来表示该patch在图片中的位置信息。ViT模型主要借鉴的是前面所说的Bert模型，它在Position Embedding的具体转换方法上同样交给模型去自主学习。

### Extra learnable [class] embedding

在ViT中作者加入了一个额外的可学习的嵌入向量，它用于表示整个图像的类别信息，即ViT架构图中的“0\*”。它的目的是从其他图像块的嵌入向量中学习到全局的特征，作为图像的表示向量。这类似Bert模型中的[CLS]token，都是用于获取整个序列的信息。

### Compared with CNN

相比于CNN，ViT有一些常见的优点，例如利用无标注数据的能力，更好的并行性，泛化能力更强，鲁棒性更好。这些优点有些也来自transformer模型，在前文中已提及，这里就不再展开。而ViT也不仅仅都是优点，当训练数据集不够大的时候，ViT的表现通常比同等大小的ResNets要差一些，这是因为CNN在CV领域已经得到了成熟的应用，利用transformer难以使用CNN的一些先验知识，如locality/two-dimensional neighborhood structure（图片的二维关系）和translation equivariance（平移不变性）。但当数据集不断扩大时，ViT在多项任务上的性能反超了ResNets。在论文中，作者也提到了将ResNets与transformer结合进行训练（Hybrid），即在图片Embedding时借助ResNets，将图片转化为向量。Hybrid在小数据集上的表现均好于ViT和ResNets，但在大数据集上与却差于ViT。这说明Transformer在CV领域也有取代CNN的势头。

![模型性能比较](模型能力对比图.png)

## 总结

在阅读了大量的关于transformer论文后，我们不难发现transformer在广度上不断扩展，从提出之时的机器翻译到NLP整个领域再到CV等其他深度学习的领域，transformer被应用在各种不同的任务中，或与传统RNN和CNN结合，或直接开辟一条新的方法，为不同领域不同任务的解决提供了更多的选择。而在深度上，从时间线上来看，transformer模型也在不断的改善和调整来适应不同的任务。包括在数据量和模型参数量上的不断扩大，将预训练和微调的思想加入transformer，这些研究都在不断的提高transformer模型的学习能力和泛化能力。

下面我列举一下transformer模型目前在哪些领域已取得了比较大的成功
- 自然语言处理： Transformer 及其变体已在 NLP 任务中得到广泛探索和应用，例如机器翻译、语言建模和命名实体识别。大量的努力致力于在大规模文本语料库上预训练 Transformer 模型，这是 Transformer 在 NLP 中广泛应用的主要原因之一。

- 计算机视觉： Transformer 还适用于各种视觉任务，例如图像分类、物体检测、图像生成和视频处理。

- 音频应用： Transformer 还可以扩展到与音频相关的应用，例如语音识别，语音合成，语音增强和音乐生成 。

- 多模态应用：由于其灵活的架构，Transformer 还被应用于各种多模态场景，例如视觉问答、视觉常识推理、字幕生成、语音到文本翻译和文本到图像生成。



### 个人思考

目前关于transformer的研究的应用仍在不断发展。例如在CV领域的swim transformer模型基于ViT模型进行改进，它的主要特点是使用滑动窗口机制，将图像分成多个小块，然后在每个小块内进行自注意力计算，这样可以降低计算复杂度，同时也保留了局部信息。除此之外，Transformer打破NLP和CV研究的鸿沟，为多模态的应用带来了希望。如今我们已经可以用上很多基于Transformer的多模态的应用，例如通过语言描述生成代码、图片、视频、音乐等等，这些应用极大的促进了AIGC，AGI的发展。

因此在未来的研究中，我们可以进一步改进Transformer，如对Transformer优秀的性能进行理论分析；对Transformer的学习和生成过程进行解释，减少黑箱性；对Transformer的模内和跨模态注意力的设计进行改进。

相信在众多的研究者的努力下，Transformer的能力可以得到进一步的发展，Transformer可以被运用到更多的领域上，更好的改善我们的生活。



## 参考文献

- [1] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin,  “Attention is all you need,” *Advances in neural information processing systems*, vol.30, 2017.

- [2] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” *arXiv preprint arXiv:1810.04805*, 2018.

- [3] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” *arXiv preprint arXiv:2010.11929*, 2020.

- [4] A. Radford, K. Narasimhan, T. Salimans, I. Sutskever *et al.*, “Improving language understanding by generative pre-training,” 2018.

- [5] A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*, “Language models are unsupervised multitask learners,” *OpenAI blog*, vol.1, no.8, p.9, 2019.

- [6] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell *et al.*, “Language models are few-shot learners,” *Advances in neural information processing systems*, vol.33, pp.1877–1901, 2020.

- [7] B. McCann, N. S. Keskar, C. Xiong, and R. Socher, “The natural language decathlon: Multitask learning as question answering,” *arXiv preprint arXiv:1806.08730*, 2018.

- [8] N. Parmar, A. Vaswani, J. Uszkoreit, L. Kaiser, N. Shazeer, A. Ku, and D. Tran, “Image transformer,”  *International conference on machine learning*. PMLR, pp.4055–4064, 2018.

- [9] T. Lin, Y. Wang, X. Liu, and X. Qiu, “A survey of transformers,” *AI Open*,2022.

- [10] Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin transformer: Hierarchical vision transformer using shifted windows,” *Proceedings of the IEEE/CVF international conference on computer vision*, 2021, pp.10012–10022.

- [11] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark *et al.*, “Learning transferable visual models from natural language supervision,” *International conference on machine learning*. PMLR, pp. 8748–8763, 2021.

- [12] H. Chefer, S. Gur, and L. Wolf, “Transformer interpretability beyond attention visualization,” *Proceedings of the IEEE/CVF conference on computer vision and pattern recognition*, pp.782–791, 2021.