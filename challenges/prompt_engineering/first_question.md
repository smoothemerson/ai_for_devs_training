- Prompt utilizado:

```
<CONTEXTO>
Aja como um expert em algoritmos. Seu objetivo é devolver uma resposta para um usuário que não entende algoritmos de forma clara, objetiva e breve.

<TAREFA>
Elabore uma explicação clara para um leigo.

<REQUISITOS>
1. Explicação clara e objetiva, a resposta não deve ser longa e deve ser clara sem complexidade.
2. Explicação organizada de forma que se tiver listagens, use bullet points.

<PERGUNTA>
Como o algoritmo de Ray Tracing calcula a cor de um pixel em uma imagem renderizada.
```

- Resposta obtida:
```
Como o Ray Tracing calcula a cor de um pixel
O Ray Tracing simula como a luz funciona no mundo real, mas de trás para frente:
O processo básico:

Lança um raio - Da câmera através de cada pixel da tela para dentro da cena 3D
Detecta colisões - Verifica onde esse raio atinge objetos (qual superfície ele bate primeiro)
Calcula iluminação - No ponto de impacto, lança novos raios em direção às fontes de luz para ver se estão bloqueadas (sombras)
Soma as contribuições - Combina a cor do objeto + luz que chega nele + reflexos (lançando mais raios nas direções de reflexo/refração)
Define a cor final - Todo esse cálculo resulta na cor que você vê naquele pixel

Em resumo: É como perguntar "o que meu olho veria olhando nessa direção?" e então rastrear matematicamente o caminho da luz até descobrir a resposta.
Por isso gera imagens realistas mas é computacionalmente pesado - cada pixel exige múltiplos cálculos de raios ricocheteando pela cena.
```
