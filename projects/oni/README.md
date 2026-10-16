# Oni — personagem jogável (MMORPG isométrico, estilo Hades + pixel art)

Personagem feita a partir das referências: pele vermelha com listras de tigre, cabelo branco
longo com reflexos iridescentes (ciano, rosa, lilás), chifres vermelho → branco, olhos brancos
com pupila em cruz, orelhas pontudas, enfeite de floco de neve prateado do lado esquerdo e
vestido branco drapeado (ombros à mostra, mangas caídas nos braços, faixa na cintura e saia longa
com fendas na frente e atrás).

Gerada por código: [`build.py`](build.py) (personagem) e [`armor_sample.py`](armor_sample.py) (armadura de teste).

## Arquivos (`exports/`)

| Arquivo | Conteúdo |
|---|---|
| `oni.fbx` | **Arquivo principal**: esqueleto, malhas, materiais, texturas e animações Idle/Walk/Run |
| `oni.glb` | Mesmo conteúdo em glTF binário (web, Godot, visualizadores) |
| `oni_palette.png` | Paleta de 21 cores (textura minúscula, filtro *nearest*); troque as cores para criar variações |
| `oni_turnaround.png`, `oni_portrait.png` | Frente / 3-4 / lado / costas e retrato |
| `oni_ingame_pixel.png` | Prévia na câmera isométrica com pixelização e contorno (como no jogo) |
| `oni_Walk.png`, `oni_Run.png`, `oni_Idle.png` | Quadros das animações |
| `oni_viewer.html` | Visualizador 3D interativo |
| `armor_sample/oni_armor_sample.fbx` | Armadura de teste (peitoral, ombreiras, braçadeiras, katana) no mesmo esqueleto |
| `armor_sample/oni_with_armor.png` | A personagem vestindo a armadura de teste |

## Especificações

- Altura 1,70 m (1,89 m com os chifres). Escala 1 unidade = 1 metro. Eixo Y para cima, frente em +Z.
- Pose de repouso: **T-pose**. Pés em y = 0.
- ~29,8 mil triângulos no total (corpo modular ~13 mil, cabelo ~6,6 mil, roupa ~8,4 mil, chifres/olhos/enfeite ~1,9 mil).
- 3 materiais: `Oni_Palette` (pele, cabelo, chifres e roupa), `Oni_Eyes` (textura 16×16 pixel art) e `Ornament` (metal).

## Preparada para armaduras e equipamentos

**Esqueleto (32 ossos)** com nomes do padrão Humanoid do Unity, que são mapeados automaticamente:
`Hips, Spine, Chest, Neck, Head, Left/RightShoulder, Left/RightUpperArm, Left/RightLowerArm, Left/RightHand,
Left/RightUpperLeg, Left/RightLowerLeg, Left/RightFoot, Left/RightToes`.

Além dos ossos do corpo, há **cadeias de ossos de cabelo** (`Hair_Back_1..5`, `Hair_Front_L_1..3`,
`Hair_Front_R_1..3`) para física de cabelo na engine (Spring Bone, Dynamic Bone, Magica Cloth…).

**Sockets** (objetos vazios presos a ossos). Para equipar, coloque o objeto como filho do socket:

| Socket | Osso | Uso |
|---|---|---|
| `Socket_Weapon_R` / `Socket_Weapon_L` | RightHand / LeftHand | armas na mão |
| `Socket_Shield_L` | LeftLowerArm | escudo |
| `Socket_Back` | Chest | arma ou mochila nas costas |
| `Socket_Head` | Head | elmo, chapéu, coroa |
| `Socket_Hip_L` / `Socket_Hip_R` | Hips | bainha, bolsa, adaga |

**Corpo modular** (todas as partes usam o mesmo esqueleto, com normais contínuas, então as junções não
aparecem): `Body_Head`, `Body_Torso`, `Body_Arms`, `Body_Hands`, `Body_Legs`, `Body_Feet`.
Ao equipar uma armadura, esconda as partes cobertas por ela para evitar que atravessem a armadura.

**Peças separadas**: `Outfit` (roupa) e `Hair` têm skinning; `Horns`, `Eye_L`, `Eye_R` e
`HairOrnament` são filhos rígidos do osso `Head`. Por exemplo, um elmo pode esconder o cabelo e os chifres.

### Armadura com skinning (peças que deformam)

Modele a peça sobre a personagem em T-pose e faça o skinning nos **mesmos nomes de ossos**.
O arquivo `armor_sample/oni_armor_sample.fbx` é um exemplo pronto. Na engine:

- **Unity**: instancie a armadura e, para cada `SkinnedMeshRenderer`, troque `bones[]` pelos
  transforms de mesmo nome do esqueleto da personagem e use o mesmo `rootBone`.
  Essa é a técnica padrão de troca de equipamento.
- **Unreal**: use o mesmo Skeleton asset e *Leader Pose Component* (Set Leader Pose).
- **Godot**: coloque a `MeshInstance3D` da armadura sob o mesmo `Skeleton3D` (`skeleton = ..`).

## Visual pixelado no estilo Hades

- Todas as cores vêm de uma **paleta de 21 cores** em `oni_palette.png`. Cada face usa uma cor só, o
  que dá o visual de pixel art. Use **filtro Point/Nearest e sem compressão** nessa textura.
- **Variações de cor**: edite `oni_palette.png`, por exemplo pele azul ou cabelo preto, sem tocar no modelo.
- Na engine, use **shader toon** (2 ou 3 faixas de luz), **contorno** e **luz de borda (rim light)**.
  Para o aspecto pixelado, renderize a cena em resolução baixa (render texture com *point filter*) e
  amplie com pixels inteiros. É assim que a prévia `oni_ingame_pixel.png` foi feita.
- A saia tem fendas na frente e atrás: cada painel acompanha uma perna, então correr não rasga a saia.

## Importação

- **Unity**: Model → Scale Factor 1 com *Convert Units* marcado; Rig → **Humanoid**, Avatar
  "Create From This Model". Nas texturas: Filter Mode = **Point**, Compression = None.
- **Unreal**: Import Uniform Scale 1; *Convert Scene* ligado; Skeletal Mesh com animações.
- **Blender / Godot**: importam o FBX ou o GLB diretamente (verificado no Blender: armature, partes,
  sockets e as 3 animações).

## Limitações desta versão

- A roupa e o cabelo não têm física; os ossos de cabelo estão prontos para receber spring bones.
- Mãos simplificadas, sem ossos de dedos.
- As animações incluídas (Idle, Walk, Run) são de referência; o rig Humanoid aceita animações
  redirecionadas, como Mixamo ou as da Unity Asset Store.
- Cabelo e roupa têm alguns poucos vértices "non-manifold" gerados pela malha implícita (7 e 24 arestas);
  não afetam engines de jogo, mas aparecem se você rodar uma checagem de malha estrita.
- Não há LODs automáticos ainda; se precisar de versões mais leves, peça e eu gero com
  resolução menor (`python projects/oni/build.py 80`).
