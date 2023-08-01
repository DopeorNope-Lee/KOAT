# K(G)OAT
IA3방식으로 KoAlpaca를 fine tuning한 한국어 LLM모델

요즘 LLM은 대부분 LORA방식으로 훈련이 진행되고 있으며, 최근은 양자화 방식을 도입한 QLORA방식으로 많이 훈련되고 있습니다.

KOAT base모델인 KoAlpaca 또한, QLORA방식으로 메모리를 효율적으로 활용하는 방법을 소개하고 있습니다.

  - 자세한 내용은 Beomi 님의 https://github.com/Beomi/KoAlpaca 깃허브를 참조바랍니다.

저희 연구는 정철현 박사님의 주도하에 연구를 진행되었으며 빠른 훈련방식과 더 효율적인 방법론으로 소개된 IA3(T-Few) 방식으로 fine-tuning진행하여,
더 적은 파라미터로 더 효율적인 훈련방식을 도입하였습니다.
  - PEFT에서는 현재 IA3방식을 implement할 수 있으나, 주의할점은 Casual LM task에서는 라이브러리 코드를 수정해야 합니다.
---
# K(G)OAT의 훈련 방식

- baseline은 Beomi님의 KoAlpaca코드를 참조하였습니다.
- 양자화 방식은 사용되지 않았습니다.
  -- (Few shot learning 평가시, 모델을 비트 단위로 불러오게 되면, Fewshot러닝 평가가 되지 않는 이슈가 있습니다.)

