# Niryk — protótipo 2: personagens (Pyram3D)

Modelos lisos para câmera próxima (sobre o ombro e 1ª pessoa). Tudo é gerado por script:
não edite os GLBs à mão.

## Corpos

`human_m`, `human_f` (humanos anime) e `harpin_m`, `harpin_f` (Harpins: asas enormes dobradas nas
costas, pernas de águia digitígradas cobertas de plumas com garras, cauda longa emplumada, penas nas
têmporas, orelhas pontudas). Todos saem de `build_characters.py`.

## Comandos

```bash
# Todos os corpos (ou só alguns), em processos paralelos
python projects/niryk/protótipo_2/build_characters.py [human_m] [human_f] [harpin_m] [harpin_f] [--fast] [--jobs 2]
#   --fast: malha mais grossa (iteração); as exportações usam o padrão
#   --no-review, --no-clips, --all-clips, --no-cache

# Teste sem tocar nos GLBs que o jogo importa: exportações e revisões vão para outra pasta
NIRYK_P2_OUT=/tmp/p2 python projects/niryk/protótipo_2/build_characters.py human_f --fast
```

`build_human.py` continua existindo (= `build_characters.py human_m`). As malhas lentas (roupa, pele,
asas e cauda dos Harpins) ficam em cache por etapa (`.cache/`, chave = hash do que cada etapa usa):
mexer no rosto, no cabelo ou na pintura não refaz a roupa. `--no-cache` refaz tudo.

Saídas:

- `exports/characters/<corpo>.glb` — corpo, roupa inicial, rosto, os penteados, rig e clipes.
- `exports/equipment/sword_iron.glb` — espada de ferro.
- `review/characters/<corpo>_*.png` — giro (toon), rosto (`_face.png`: de frente, 3/4 e perfil com luz
  da esquerda, de frente e de trás-esquerda, a pele sombreada pelo mapa de sombra como o cliente
  desenha), rosto ao lado das referências (`_face_ref.png`: de perto e à distância de jogo), penteados,
  paletas, mãos, tiras de animação; `<corpo>_stats.json` com triângulos por parte.

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
6. **Cabeça** (`head.py`, linha Genshin): crânio grande e um tubo de perfil para o rosto (largura =
   silhueta de frente: queixo pequeno e pontudo, bochechas cheias; frente = perfil: testa, linha dos
   olhos um pouco recuada, base do nariz, boca, queixo; seção mais quadrada nos olhos, então de frente
   o rosto lê liso). Volumes finos por deslocamento suave do SDF (`sculpt_bumps`: área dos olhos
   levemente recuada, bochechas, queixo), nariz = pontinha pequena. Feminino mais redondo e jovem;
   masculino com o terço inferior mais longo e a mandíbula mais firme. Sem geometria de olho/boca.
7. **Rosto pintado** (`face.py`, `face_paint.py`): olhos, cílios, sobrancelhas e boca são texturas
   pintadas com numpy (formas com antisserrilhado e degradês, 1024 px) em **decalques**: cópias dos
   triângulos do `Head_Skin` de cada região, empurradas 0,4 a 1 mm pela normal (os opacos — branco
   do olho e íris — recortados no contorno). Olho grande (~1/6 da altura do rosto): íris escura em
   cima e clara embaixo, aro, pupila, dois ou três brilhos, linha grossa de cílios que se abre no canto
   externo (cílios no feminino), prega da pálpebra, sombra da pálpebra no branco, sugestão de cílio
   inferior; sobrancelha fina, boca mínima, rubor na pele. O olho direito espelha o esquerdo.
   **Sombra do rosto por mapa** (técnica do Genshin): veja "Contrato do rosto" abaixo.
   **Cabelo** (`hair.py`): calota lisa de volume + mechas finas e pontudas em camadas, cada uma um
   varrido de seção em lente (larga e fina, bordas curvadas para a cabeça) numa gaiola grossa
   suavizada por um passo de Catmull-Clark; normais da superfície lisa, puxadas em parte para um casco
   em volta da cabeça (faixas de toon limpas). Franja emoldurando o rosto (sobre as íris as pontas
   param acima da linha de cílios), mechas laterais na frente das orelhas, massa lisa atrás. Dois tons
   nas cores de vértice (base 0,8; lado de baixo / de dentro 0,6) e o anel de anjo (1,0), uma faixa em
   zigue-zague cortada na malha para ter borda nítida.
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

  | Nó | Conteúdo | Material |
  |---|---|---|
  | `Body` | pele abaixo do pescoço (braços, mãos) | `Skin` (sem texturas) |
  | `Head_Skin` | cabeça e pescoço, com UVs | `Skin` com `baseColorTexture` + `occlusionTexture` (mapa de sombra) |
  | `Head_Eyes` | brancos dos olhos | `EyeWhite` |
  | `Head_Iris` | íris e pupilas | `Eye` |
  | `Head_EyeShine` | brilhos dos olhos | `EyeShine` (MASK) |
  | `Head_Lashes` | linha de cílios, cílios, prega da pálpebra, cílio inferior | `Lash` (MASK) |
  | `Head_Brows` | sobrancelhas | `Brow` (MASK) |
  | `Head_Mouth` | boca | `Mouth` (MASK) |
  | `Head_Plumes` | Harpins: penas das têmporas | `Plume` |
  | `Hair_curto` … | um penteado por nó (o cliente mostra um) | `Hair` (+ `Leather` nos prendedores) |
  | `Outfit` | túnica, calça, cinto + fivela, braçadeiras, botas | `Cloth_A`, `Cloth_B`, `Leather`, `Metal` |

  Em 1ª pessoa, esconda os nós que começam com `Head_` ou `Hair_`.
- **Slots de cor**: o cliente multiplica a cor escolhida. As cores base dos slots
  (`Skin, Hair, Eye, Brow, Cloth_A, Cloth_B, Leather, Metal`) são brancas; o detalhe pintado
  (bainhas, solas, tiras) está na luminância das cores de vértice e, no rosto, nas texturas.
  `EyeWhite`, `EyeShine`, `Lash` e `Mouth` não são tingidos; `Brow` leva a cor do cabelo. Tintas padrão do visual inicial (usadas nas revisões, `common.TINTS`):
  Skin `#e8b48e`, Hair `#5e3a24`, Eye `#3f8a5c`, Cloth_A `#4d7d3c`, Cloth_B `#6e5a48`,
  Leather `#7c5032`, Metal `#b9c0ca`.

## Contrato do rosto (cliente)

- **Materiais**: `Skin` do `Head_Skin` (tingido pela pele; `baseColorTexture` quase branca com rubor e
  um toque sob o nariz; `occlusionTexture` = mapa de sombra, abaixo). `EyeWhite` (sem tinta), `Eye`
  (íris em cinza, tingida pela cor dos olhos), `EyeShine` (branco, sem tinta, `MASK`), `Lash` (marrom
  escuro/preto, sem tinta, `MASK`), `Brow` (cinza, tingido pela cor do cabelo, `MASK`), `Mouth` (sem
  tinta, `MASK`). Todos os `MASK` com `alphaCutoff` 0,5 (o alfa tem uma rampa de ~1,6 px na borda, então
  o contorno a 0,5 fica liso de perto). O `Body` usa outro material também chamado `Skin`, sem texturas.
- **Decalques**: `Head_*` rígidos no osso `Head`, 0,4 a 1 mm acima da pele (branco 0,42, íris 0,56,
  sobrancelha e boca 0,51, brilho 0,74, cílios 0,88 mm no feminino; 0,45 / 0,6 / 0,55 / 0,8 / 0,95 mm no
  masculino). Normais =
  as da pele.
- **UVs do `Head_Skin`** (`head.py`): u = azimute em volta de um eixo vertical (u = 0,5 na linha do
  meio do rosto, o lado **esquerdo** do personagem em u > 0,5, costura descendo pela nuca, a textura
  repete em u), com mais resolução na frente; v = elevação vista da junção cabeça-pescoço. Simétrico:
  u(−x) = 1 − u(x). Mesmas UVs para a cor e para o mapa.
- **Mapa de sombra do rosto** (`occlusionTexture`, linear, canal R; 1024 px): autorado para luz vinda
  da **esquerda** do personagem (+X). Seja φ o ângulo horizontal entre a frente da cabeça (+Z) e a
  direção da luz (espaço da cabeça, projetado em XZ), de 0 (de frente) a π (bem de trás) passando
  pelo lado esquerdo. O texel com valor `t` está **iluminado enquanto φ < t·π** (então a bochecha
  direita escurece primeiro e a sombra varre o rosto). Para luz vinda da direita, o cliente espelha
  u (amostra em `1 − u`). Base: no rosto, um modelo redondo por altura (t = ½ + asen(x / w(y)) / π, w =
  meia largura do rosto), então a fronteira segue a curva da bochecha; no resto (laterais, nuca,
  pescoço), o azimute da normal de um proxy liso da cabeça (t = ½ + a/π, limitado: a metade direita
  de trás lê 0, a esquerda 1). Direção de arte: uma pequena sombra ao lado do nariz em ângulos médios
  (cresce para longe do nariz) e o **pescoço sob a mandíbula sempre na sombra** (t = 0), com borda
  nítida ao longo da mandíbula e da base do crânio.

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
- `head.py` — cabeça (SDF) e UVs do `Head_Skin`. `face.py` — decalques do rosto, UVs, materiais.
  `face_paint.py` — desenho 2D e texturas pintadas do rosto, mapa de sombra. `hair.py` — penteados.
  `clothes.py` — roupa inicial.
- `meshing.py` — SDF → malha lisa no orçamento (regiões, faces escondidas). `rig.py` — pesos.
  `anim.py` — clipes.
- `equipment.py` — espada de ferro. `build_characters.py` — monta, exporta; `review_sheets.py` —
  folhas de revisão.
- `nature.py` — natureza da clareira (outro autor).

## Harpins (`harpin_m`, `harpin_f`)

Mesmo pipeline e mesmo rosto anime dos humanos, com:

- **Asas** (nó `Wings`, slot `Plume`, ossos `Left/RightWing1..3` filhos do `Chest`): dobradas nas
  costas, o punho na altura do alto da cabeça e as pontas perto dos tornozelos. Braço da asa liso
  (SDF) e penas como **sólidos finos fechados** (seção em lente com a raque elevada e mais escura;
  funcionam com material de um lado só). As fileiras (primárias, secundárias + terciárias, grandes e
  pequenas coberteiras) se sobrepõem como telhas: cada pena é inclinada alguns graus no próprio eixo,
  então vizinhas nunca se tocam. A dobra fica numa superfície curva (`common.wing_point`: inclinada
  para trás embaixo, envolvendo o corpo para fora). Cada pena é rígida no seu osso (primárias na mão,
  secundárias no antebraço, terciárias no úmero; as internas misturam um pouco o osso vizinho).
- **Pernas de águia** (no nó `Body`): coxa sob a túnica, canela com volume de plumas terminando em
  duas franjas de plumas pontudas (a de baixo abre sobre o calcanhar), tarso fino emplumado com um
  punho de plumas sobre os dedos; 3 dedos para a frente e 1 para trás com escamas pintadas (`Skin`)
  e garras curvas (`Talon`). Ossos: `UpperLeg` = fêmur, `LowerLeg` = canela, `Foot` = tarso,
  `Toes` = dedos da frente.
- **Cauda** (nó `Tail`, slot `Plume`, ossos `Tail1..6` a partir do `Hips`): longa, coberta de plumas
  em telhas, com um pincel de mechas grandes que termina em pontas.
- **Cabeça**: o mesmo rosto pintado, orelhas pontudas (`Head_Skin`), tufos de penas nas têmporas
  (`Head_Plumes`, `Plume`), penteados `Hair_espetado`, `Hair_crista`, `Hair_plumado` (mechas anime e penas
  no slot `Hair`).
- **Roupa**: a mesma túnica e gola com fendas para as asas, mais curta atrás com um entalhe sobre a
  cauda; sem calça nem botas.
- **Clipes**: os mesmos 10, com passos digitígrados (os dedos plantam e não deslizam; o tarso rola
  sobre a base dos dedos; as garras se fecham no ar) e camadas de asas e cauda: respiração no `Idle`,
  asas um pouco abertas e para trás ao andar e correr (livram as pernas), abertura no golpe, abertas
  no `Hit`, meio abertas no `Death` (que cai de frente). Durações dos Harpins: `Walk` 0,76 s, `Run`
  0,52 s, `WalkBack` 0,8 s, `StrafeLeft/Right` 0,68 s (passos curtos e rápidos; as velocidades são
  as mesmas: o cliente toca o clipe pela velocidade real).
- Revisão: `review/characters/harpin_*_wings.png` (asas no Idle, Run, golpe, Hit, Death; pernas de
  perto), além das folhas dos humanos.
