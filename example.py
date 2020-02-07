
ncol = 8
nrow = 8
edges = []
size = ncol*nrow
for i in range(1, size, 1):
        res = i%ncol
        print("res", i, res)

        if i < ncol:
                edges.append({'data': {'source': i, 'target': i + 1}})
                edges.append({'data': {'source': i, 'target': i + ncol}})
                edges.append({'data': {'source': i, 'target': i + (ncol + 1)}})

        elif (res != 0 and i > ncol) and i < ncol*(nrow-1):
            edges.append({'data': {'source': i, 'target': i + 1}})
            edges.append({'data': {'source': i, 'target': i - (ncol - 1)}})
            edges.append({'data': {'source': i, 'target': i + ncol}})
            edges.append({'data': {'source': i, 'target': i + ncol + 1}})

        elif res == 0:
                edges.append({'data': {'source': i, 'target': i + ncol}})

        elif i > ncol*(nrow-1):
            edges.append({'data': {'source': i, 'target': i + 1}})

