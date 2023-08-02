<p align="center" width="100%">
</p>

# K(G)OAT🐐
- IA3방식으로 KoAlpaca를 fine tuning한 한국어 LLM모델

- 요즘 LLM은 대부분 LoRA방식으로 훈련이 진행되고 있으며, 최근은 양자화 방식을 도입한 QLORA방식으로 많이 훈련되고 있습니다.

- KOAT base모델인 KoAlpaca 또한, QLORA방식으로 메모리를 효율적으로 활용하는 방법을 소개하고 있습니다.
  - 자세한 내용은 Beomi 님의 깃허브를 참조바랍니다. (https://github.com/Beomi/KoAlpaca)

- 저희 연구는 주식회사 마커의 대표이신 [정철현] 박사님의 주도하에 연구를 진행되었으며 빠른 훈련방식과 더 효율적인 방법론으로 소개된 IA3(T-Few) 방식으로 fine-tuning진행하여,
더 적은 파라미터로 더 효율적인 훈련방식을 도입하였습니다.

  - PEFT에서는 현재 IA3방식을 implement할 수 있으나, 주의할점은 Casual LM task에서는 라이브러리 코드를 수정해야 합니다.
  - 수정사항은 이슈에 남겨주시면 수정하도록 하겠습니다.

---
# K(G)OAT의 훈련 방식

- baseline은 Beomi님의 KoAlpaca코드를 참조하였습니다.😃

- 양자화 방식은 사용되지 않았습니다.
        > Few shot learning 평가시, 모델을 비트 단위로 불러오게 되면, Fewshot러닝 평가가 되지 않는 이슈가 있습니다.

- KoAlpaca 5.8b가 베이스 모델이며, Tokenizer는 Polyglot5.8b의 Tokenizer를 사용하였습니다.
  
- IA3방식이란?
  - LoRA는 rank decomposition 행렬을 사전학습된 모델에 추가하여 중간마다 학습이 가능한 파라미터를 삽입한 방식입니다.
    
  - IA3는 Infused Adapter by Inhibiting and Amplifying Inner Activations의 약자로, LORA와 비슷하게 적은 파라미터를 사전학습된 모델에 추가하여 훈련하는 방법입니다.
    
  - LoRA와의 차이점은, LoRA는 hidden state에 새로운 값을 더해주는 기법이지만, IA3의는 Attention에 Key, Value 값을 rescale해주는 벡터와 position-wise feed-forward network의 output을 rescale 하는 벡터를 추가해 훈련시키는 방식입니다.
    
  - IA3방식은 LoRA보다 적은 파라미터로 더 좋은 성능을 낸다는 방법론으로 소개되었으며 저희는 K(G)OAT를 활용하여 훈련을 진행하였습니다.

- K(G)OAT는 4epoch, maxstep 32,000 step으로 훈련이 되었으며, 총 소요된 시간은 16시간 소요되었습니다.

- 같은 방식으로 비트단위로 훈련시키지 않은 LORA방식을 적용한 KoAlpaca는 xxx 시간이 소요되었습니다.

- 훈련한 파라미터 수는 다음과 같습니다.

모델명 | 파라미터수 | 훈련 소요시간
-- | -- | --
koalpaca-polyglot | 800000 | 158 min
**K(G)OAT-polyglot** | **3000** | **158 min**
    
- K(G)OAT는 A5000 2장으로 훈련되었으며, 한동대학교 [Xiaopeng Yang](https://www.researchgate.net/profile/Xiaopeng-Yang-2) 교수님의 AIMV 연구실에서 훈련되었습니다.

---
# Dataset

## 훈련데이터셋

- 훈련에 Dataset은 기본적으로 KoAlpaca와 성능 비교를 위해 Beomi님의 KoAlpacav1.1 데이터셋을 활용하였습니다.
  
- 하지만 프롬프트 구성에 대한 수정사항은 다음과 같습니다

```python
# 기존의 코드
data = data.map(
    lambda x: {'text': f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" }
)

# 수정된 프롬프트 코드
data = data.map(
    lambda x: {'text': f"{x['instruction']}\n\n정답: {x['output']}<|endoftext|>"}
)
```

## Fewshot Learning 평가 데이터셋

- 네이버 영화리뷰 데이터셋인 NSMC 데이터를 활용하였습니다.

```json
# 기존의 코드
data = data.map(
    lambda x: {'text': f"### 질문: {x['instruction']}\n\n### 답변: {x['output']}<|endoftext|>" }
)

# 수정된 프롬프트 코드
data = data.map(
    lambda x: {'text': f"{x['instruction']}\n\n정답: {x['output']}<|endoftext|>"}
)
```


# Fewshot Learning 평가

- Fewshot Learning 평가

모델명 | 프롬프트1 | 프롬프트2
-- | -- | --
koalpaca-polyglot | 800000 | 158 min
**K(G)OAT-polyglot** | **3000** | **158 min**
    
