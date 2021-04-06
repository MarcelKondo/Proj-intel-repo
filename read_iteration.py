import panda as pd

data4 = pd.read_csv("Iteration.csv", index_col = 0)
ax4 = data4.plot(rot=0, title = 'S0 = [256 256 256 32 10 32 32 32] équal, -np 4, ngbr = Others')
ax4.legend(['Opt Speed (GFlops)', 'Opt Execution time (s)', 'Avg Speed (Gflops)', 'Avg Execution time (s)'])

fig4 = ax4.get_figure()
fig4.savefig('Iteration_figure.png')
