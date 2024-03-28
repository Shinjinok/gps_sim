import georinex as gr
import approx

nav = gr.load('brdc0730.24n').to_dataframe().dropna(how='all')

print(nav[3 'cic'])