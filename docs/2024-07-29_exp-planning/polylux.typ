#import "@preview/polylux:0.3.1": *
#import themes.clean: *


#show: clean-theme.with(
//  footer: [Andrea Pierré, Brown University],
  short-title: [DRL experiments plan],
//  logo: image("dummy-logo.png"),
)
#set text(font: "Inria Sans", size: 20pt)

#title-slide(
  title: [DRL experiments plan],
//  subtitle: [Presentation subtitle],
  authors: ([Andrea Pierré]),//, [Author B], [Author C]),
  date: [July 29, 2024],
//  watermark: image("dummy-watermark.png"),
)


//#new-section-slide("The new section")

/*
#focus-slide[
  _Focus!_

  This is very important.
]
*/

#slide(title: [1) Does the network learn a coordinate system?])[
    #side-by-side(gutter: 1em, columns: (1fr, 1fr))[
        - Redundant spatial input? Only Cartesian/polar input?
        - Expected #sym.arrow Same performance on the discretized version with zero shot learning
        - Expected #sym.arrow Discretized policy looks similar
    ][
      #image("img/coord-sys-exp.drawio.svg")
    ]
]

#slide(title: [2) How the constraints of the task impact the representations learned?])[
    #side-by-side(gutter: 1em, columns: (0.7fr, 1fr))[
        #align(horizon)[#image("img/RL_env-cartesian-polar.drawio.svg")]
    ][
        #align(horizon)[#image("img/cartesian-polar-exp-activity-heatmap.png")]
    ]
]

#slide(title: [3) Does having redundant spatial input make the agent more robust in a noisy environment?])[
    #side-by-side(gutter: 1em, columns: (1fr, 1fr))[
        - Conflicts with experiment 2?
        - Train with noise?
        - May need another architecture to solve this task (Generative Adversarial Network? Denoising Autoencoder?)
        - Expected #sym.arrow Robust but degraded performance
    ][
      #image("img/exp3.svg")
    ]
]
