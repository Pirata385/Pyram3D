# Pyram3D

Software de modelagem 3D **gratuito e de código aberto, feito para ser usado por inteligências artificiais**.
Você descreve o que quer; o agente (por exemplo o Claude Code, inclusive pelo celular) modela em código,
confere o resultado em imagens renderizadas, ajusta e entrega os arquivos **.glb**, **.gltf** e **.fbx**
prontos para Unity, Unreal, Godot, Blender, three.js e outros.

Não é preciso instalar nada na sua máquina: tudo roda em Python + numpy dentro da sessão do agente.
É uma alternativa ao "Blender MCP" sem depender do Blender.

## Como usar (pelo Claude Code)

Abra uma sessão do Claude Code neste repositório e peça, por exemplo:

> Crie uma casa de fazenda low-poly com celeiro e cerca, para usar no Unity. Exporte em FBX e GLB.

> Faça um personagem estilizado de 1,70 m com camiseta verde, com animações de andar, correr e pular.

O agente segue o fluxo descrito em [`CLAUDE.md`](CLAUDE.md): cria `projects/<nome>/build.py`, gera prévias
(`_preview.png`, `_views.png`), corrige o que estiver errado e salva em `projects/<nome>/exports/`:

| Arquivo | Para quê |
|---|---|
| `nome.glb` | Formato binário glTF (web, Godot, Unity com glTFast, Blender) |
| `nome.gltf` | glTF em JSON com tudo embutido |
| `nome.fbx` | FBX 7.4 binário (Unity, Unreal, Maya, 3ds Max, Blender) |
| `nome_preview.png`, `nome_views.png` | Renderizações de conferência |
| `nome_viewer.html` | Visualizador 3D interativo (abre no celular) |

## O que ele faz

- **Modelagem**: primitivas (cubo, esfera, cilindro, cone, toro, cápsula…), extrusão, inset, subdivisão
  Catmull-Clark, espelhamento, solidificar, dobrar/torcer/afunilar, revolução (lathe), varredura (sweep), loft,
  extrusão de formas 2D, solda de vértices, UVs (caixa, planar, cilíndrica, esférica).
- **Modelagem orgânica e booleanas** com SDF (união suave, subtração, interseção) → malhas fechadas.
- **Escultura digital**: pincéis draw, inflate, grab, smooth, flatten, pinch, crease e noise, com simetria.
- **Materiais PBR** (metal/rugosidade, emissão, transparência, unlit) e **texturas procedurais**
  (madeira, mármore, tijolos, azulejos, ruído, células, listras, mapa normal a partir de altura).
- **Cena**: hierarquia, luzes (direcional, ponto, spot), câmeras, instâncias.
- **Animação** por keyframes (posição, rotação, escala, blend shapes) com interpolação linear, degrau ou suave.
- **Rigging**: esqueletos, pesos automáticos, skinning; personagens humanoides e quadrúpedes prontos, com ciclos
  de andar, correr, idle, pular e acenar (nomes de ossos no padrão Humanoid do Unity).
- **Simulações** gravadas como animação (tocam em qualquer engine): corpos rígidos caindo/quicando, tecido
  caindo sobre objetos (toalhas, cobertores, bandeiras) e partículas em loop (fogo, fumaça, faíscas, chafariz,
  neve, chuva, magia).
- **Efeitos de imagem** na prévia: bloom (brilho), neblina, vinheta, exposição.
- **Mundos**: terrenos com ruído, ilhas, terraços, pintura automática (areia/grama/rocha/neve), água e
  espalhamento de objetos.
- **Geradores**: árvores, pinheiros, palmeiras, arbustos, rochas, grama, flores, cogumelos, cactos, prédios,
  casas, colunas, cercas, escadas, postes, ruas, carros (sedan/hatch/SUV/picape), barco, avião, caixas, barris,
  mesas, cadeiras, sofá, luminária, estante, espada, baú.
- **Renderização de prévia** por software (realista com sombras, toon/estilizada, argila, wireframe, normais),
  folhas de vistas e tiras de animação.
- **Importação** de .glb/.gltf/.obj para editar, renderizar ou converter (ex.: GLB → FBX).

## Uso direto em Python

```python
import pyram3d as p3

scene = p3.Scene("Mesa")
scene.add(p3.props.table())
scene.add(p3.props.chair(cushion="#aa3333"), position=(0, 0, 0.6), rotation=(0, 180, 0))
scene.add_default_lights()
p3.save(scene, "projects/mesa/exports/mesa")   # .glb .gltf .fbx + prévias + visualizador
```

Linha de comando: `python -m pyram3d info|render|convert|viewer|check|run ...`

## Estrutura

```
pyram3d/            biblioteca (mesh, primitives, sdf, sculpt, material, scene, animation,
                    armature, terrain, render, viewer, importer, export/{gltf,fbx,obj}, generators/)
projects/           modelos criados para você (um diretório por projeto)
examples/           scripts de exemplo (vila na ilha, lanterna, robô com rig próprio)
docs/API.md         referência completa da API (gerada automaticamente)
tests/              testes (python -m pytest -q)
```

## Limitações atuais

Simulação de fluidos e fumaça volumétrica, renderização por path tracing, desdobramento automático de UV,
decimação/retopologia, bevel em arestas selecionadas e IK ainda não existem. A física é simples (boa para
efeitos e poses, não para engenharia). Veja o roadmap em `docs/ROADMAP.md`.

Licença: MIT.
