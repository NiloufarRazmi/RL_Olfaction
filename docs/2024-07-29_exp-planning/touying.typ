#import "@preview/touying:0.4.2": *

#let s = themes.metropolis.register(aspect-ratio: "16-9") //, footer: self => self.info.institution)
#let s = (s.methods.info)(
  self: s,
  title: [DRL experiments plan],
  //subtitle: [Subtitle],
  author: [Andrea Pierré],
  date: [July 29, 2024],
//  institution: [Brown University],
)

#let s = (s.methods.colors)(
  self: s,
  secondary-light: rgb("#c00404"),
)
#let s = (s.methods.enable-transparent-cover)(self: s)
#(s.methods.touying-new-section-slide = none)
#let (init, slides, touying-outline, alert, speaker-note) = utils.methods(s)
#show: init

#set text(font: "Fira Sans", weight: "light", size: 20pt)
#show math.equation: set text(font: "Fira Math")
#set strong(delta: 100)
#set par(justify: true)
#show strong: alert

#let (slide, empty-slide, title-slide, new-section-slide, focus-slide) = utils.slides(s)
#show: slides.with(outline-slide: false)


/*
= First Section

#slide[
  A slide without a title but with some *important* information.
]

= Second Section

#focus-slide[
  Wake up!
]

== Simple Animation
#slide[
  A simple #pause dynamic slide with #alert[alert]

  #pause

  text.
]
*/

== 1) Does the network learn a coordinate system?
#slide(composer: (1fr, 1fr))[
    - Redundant spatial input? Only Cartesian/polar input?
    - Expected #sym.arrow Same performance on the discretized version with zero shot learning
    - Expected #sym.arrow Discretized policy looks similar
][
    #image("img/coord-sys-exp.drawio.svg")
]

== 2) How the constraints of the task impact the representations learned?
#slide(composer: (0.7fr, 1fr))[
    #image("img/RL_env-cartesian-polar.drawio.svg")

][
    #image("img/cartesian-polar-exp-activity-heatmap.png")
]

== 3) Does having redundant spatial input make the agent more robust in a noisy environment?
#slide(composer: (1fr, 1fr))[
    - Conflicts with experiment 2?
    - Train with noise?
    - May need another architecture to solve this task (Generative Adversarial Network? Denoising Autoencoder?)
    - Expected #sym.arrow Robust but degraded performance
][
    #image("img/exp3.svg")
]
