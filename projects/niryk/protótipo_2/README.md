# Niryk — protótipo 2: personagens (Pyram3D)

Modelos lisos para câmera próxima (sobre o ombro e 1ª pessoa). Tudo é gerado por script:
não edite os GLBs à mão.

## Comandos

```bash
# Humano masculino + espada de ferro (GLBs, estatísticas e folhas de revisão)
python projects/niryk/protótipo_2/build_human.py            # final (malha fina, ~5 min)
python projects/niryk/protótipo_2/build_human.py --fast     # iteração rápida (~3 min)
python projects/niryk/protótipo_2/build_human.py --no-review
python projects/niryk/protótipo_2/build_human.py --no-clips # revisão sem as tiras dos clipes

# Teste sem tocar nos GLBs que o jogo importa: exportações e revisões vão para outra pasta
NIRYK_P2_OUT=/tmp/p2 python projects/niryk/protótipo_2/build_human.py --fast
```

As malhas ficam em cache (`.cache/`, chave = hash dos scripts de modelagem); `--no-cache` refaz.

Saídas:

- `exports/characters/human_m.glb` — corpo, roupa inicial, `Hair_curto`, rig e clipes.
- `exports/equipment/sword_iron.glb` — espada de ferro.
- `review/characters/human_m_*.png` — giro (toon), close-up da cabeça/tronco (toon e clay),
  close-up das mãos (`human_m_hands.png`: mão da espada de frente, de lado, de baixo, em 1ª pessoa
  no golpe e no acerto do `Attack1`; mão relaxada de três lados), tiras de animação;
  `human_m_stats.json` com triângulos por parte.

## Como é feito (sem caroços)

1. **Formas grandes e contínuas** (`shapes.Tube`): tronco, braços, pernas, pescoço e o bloco
   rosto/mandíbula são tubos com perfis suaves (largura, profundidade frente/trás, centro e
   "quadratura" da seção variando ao longo do eixo). Um braço inteiro (deltoide → bíceps →
   cotovelo → antebraço → pulso) é uma única superfície. Uniões amplas só nas junções.
2. **Malha** (`meshing.mesh_sdf`): superfície em grade (SDF), suavização Taubin (preserva
   volume), reprojeção na superfície exata, **decimação por erro quadrático** (`Mesh.decimate`)
   com limite de aresta por região e **normais analíticas** do SDF (gravadas no glTF). A cabeça
   e as duas mãos são malhadas à parte numa grade fina (2,8 mm), o resto do corpo numa grade
   grossa (5,2 mm); os cortes entre as partes ficam escondidos sob a túnica e as braçadeiras.
3. **Nada escondido gasta triângulos**: a pele sob a roupa e sob o cabelo, as faces internas das
   roupas (tampas de corte dentro do corpo, a túnica sob o cinto, a calça sob as botas e no alto
   da saia) e o interior do cabelo são apagados antes e depois da decimação.
4. **Roupas**: cascas lisas (`shapes.Offset`) das formas do corpo, cortadas com bainhas
   arredondadas (a manga curta tem uma bainha enrolada); a saia da túnica é um tubo em A largo o
   bastante para as coxas não aparecerem através dela.
5. **Mãos**: palma larga e quadrada, dedos grossos e cônicos de pontas rombas (médio, anelar e
   mínimo fundidos num bloco com sulcos, indicador um pouco separado, polegar à parte), nós dos
   dedos marcados. A mão direita é um punho fechado no cabo da espada (os dedos contornam o eixo
   do cabo e o polegar fecha por baixo); a esquerda fica relaxada. Não há ossos de dedos.
6. **Rosto**: arco das sobrancelhas forte, maçãs do rosto, mandíbula e queixo quadrados, nariz de
   ponte reta, olhos amendoados (íris grande com anel escuro, pupila, brilho e linha de cílios),
   lábios com a linha da boca e a cavidade sob o lábio inferior.
7. **Cabelo** (`hair.py`): calota de volume coberta por mechas grossas (`shapes.Strand`, um
   varrido liso de seção lenticular): topete que sobe da testa e vai para trás, mechas largas
   saindo do cocuruto, laterais penteadas para trás e pontas curtas na nuca.
8. **Pesos** (`rig.py`): funções suaves da posição (pertinência ao braço/perna/cabeça e
   transições em smoothstep nas juntas). Corpo e roupa usam a mesma função; nada é rígido nas
   juntas. As mãos nunca recebem peso das coxas; cada lado da saia segue a sua coxa e o meio é
   dividido entre as duas. Cabelo, olhos e sobrancelhas seguem o osso `Head`.

## Convenções do GLB

- Metros, +Y para cima, frente +Z, esquerda do personagem = +X, pés em y = 0.
- **Pose de repouso: A-pose** (braços 40° abaixo da horizontal). Os clipes são autorados em
  direções de mundo, então não dependem dela.
- Ossos (Unity Humanoid, os mesmos do protótipo 1): `Hips, Spine, Chest, Neck, Head`,
  `Left/Right` + `Shoulder, UpperArm, LowerArm, Hand, UpperLeg, LowerLeg, Foot, Toes`.
- Sockets: `Socket_Weapon_R` (na `RightHand`), `Socket_Back` (no `Chest`), `Socket_Head`.
- **Espada**: origem no centro do cabo (a mão fecha em volta do eixo Y), lâmina ao longo de +Y,
  gumes em ±X, faces planas em ±Z. Vai como filha de `Socket_Weapon_R` sem deslocamento: a lâmina
  segue o eixo do polegar da mão direita e os gumes ficam alinhados com os nós dos dedos. No
  repouso a lâmina aponta para a frente (+Z). No clipe `Death` o socket é animado (a espada cai
  da mão e fica no chão); todos os outros clipes também chaveiam o socket (volta para a mão).
- Nós (o que o cliente procura):

  | Nó | Conteúdo | Materiais |
  |---|---|---|
  | `Body` | pele abaixo do pescoço (braços, mãos) | `Skin` |
  | `Head_Skin` | cabeça e pescoço | `Skin` |
  | `Head_Eyes` | globo, íris, pupila, brilho, linha dos cílios | `EyeWhite`, `Eye`, `Dark` |
  | `Head_Brows` | sobrancelhas | `Hair` |
  | `Hair_curto` | cabelo curto | `Hair` |
  | `Outfit` | túnica, calça, cinto + fivela, braçadeiras, botas | `Cloth_A`, `Cloth_B`, `Leather`, `Metal` |

  Em 1ª pessoa, esconda os nós que começam com `Head_` ou `Hair_`.
- **Slots de cor**: o cliente multiplica a cor escolhida. As cores base dos slots
  (`Skin, Hair, Eye, Cloth_A, Cloth_B, Leather, Metal`) são brancas; o detalhe pintado
  (bainhas, solas, tiras, lábios, íris) está na luminância das cores de vértice. `EyeWhite` e
  `Dark` não são tingidos. Tintas padrão do visual inicial (usadas nas revisões, `common.TINTS`):
  Skin `#e8b48e`, Hair `#5e3a24`, Eye `#3f8a5c`, Cloth_A `#4d7d3c`, Cloth_B `#6e5a48`,
  Leather `#7c5032`, Metal `#b9c0ca`.

## Clipes (30 fps, todos os ossos chaveados, LINEAR)

| Clipe | Duração | Laço | Velocidade (passada = vel. × duração) |
|---|---|---|---|
| `Idle` | 2,4 s | sim | — |
| `Walk` | 1,0 s | sim | 1,9 m/s (`WALK_SPEED`) |
| `Run` | 0,64 s | sim | 5,0 m/s (`RUN_SPEED`) |
| `WalkBack` | 1,0 s | sim | 1,33 m/s (1,9 × `BACKPEDAL_FACTOR` 0,7) |
| `StrafeLeft` / `StrafeRight` | 0,8 s | sim | 1,9 m/s para +X / −X |
| `Attack1` | 0,5 s | não | golpe horizontal (acerto em 0,2 s) |
| `Attack2` | 0,55 s | não | diagonal de revés, continua o combo (acerto em 0,24 s) |
| `Hit` | 0,36 s | não | — |
| `Death` | 1,2 s | não | — |

Os pés plantados não deslizam nessas velocidades (IK de duas juntas, pé rolando no calcanhar
ou na planta).

## Arquivos

- `common.py` — proporções, rig, sockets, slots, exportação, luzes de revisão.
- `shapes.py` — `Tube` (tubo de perfil), `Offset` (casca de roupa), `Strand` (mecha), `wedge`.
- `body.py` — corpo (tronco, ombros, pescoço, braços, mãos relaxada e fechada no cabo, pernas,
  pés, cabeça).
- `face.py` — olhos e sobrancelhas. `hair.py` — `Hair_curto`. `clothes.py` — roupa inicial.
- `meshing.py` — SDF → malha lisa no orçamento (regiões, faces escondidas). `rig.py` — pesos.
  `anim.py` — clipes.
- `equipment.py` — espada de ferro. `build_human.py` — monta, exporta e renderiza a revisão.
- `nature.py` — natureza da clareira (outro autor).
