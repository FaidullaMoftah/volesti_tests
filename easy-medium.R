test = 1 #2 for medium
library(plotly)
library(volesti)
cube = gen_cube(dimension = 3, representation = "V")

type = c('RDHR', 'BRDHR')[test]
v = as.data.frame(attr(cube, "V"))
fig = plot_ly(
  x = v[, 1], y = v[, 2], z = v[, 3],
  type = 'mesh3d',
  alphahull = 0,
  opacity = 0.5
)
s = sample_points(cube, num_points, random_walk = c(walk = type))
for (i in 1:num_points){
  fig = fig %>% add_trace(
    x = s[1, i], y = s[2, i], z = s[3, i],
    type = 'scatter3d',
    mode = 'markers',
    marker = list(size = 5, color = 'black', opacity = 1)
  )
  print(plotly_build(fig))
  Sys.sleep(2)
}
