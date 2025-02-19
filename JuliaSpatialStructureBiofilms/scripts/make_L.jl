function makeL2D_closedboundaries(n, m)
    L = fill(0, n * m, n * m)
    lindices = LinearIndices((n, m))

    for x in 1:n
        for y in 1:m
            li = lindices[x, y]
            if x != 1
                L[li, lindices[x-1, y]] += 1
                L[li, li] -= 1
            end
            if x != n
                L[li, lindices[x+1, y]] += 1
                L[li, li] -= 1
            end
            if y != 1
                L[li, lindices[x, y-1]] += 1
                L[li, li] -= 1
            end
            if y != m
                L[li, lindices[x, y+1]] += 1
                L[li, li] -= 1
            end
        end
    end

    L
end

function makeL3D_closedboundaries(n, m, k)
    L = fill(0, n * m * k, n * m * k)
    lindices = LinearIndices((n, m, k))

    for x in 1:n
        for y in 1:m
            for z in 1:k
                li = lindices[x, y, z]
                if x != 1
                    L[li, lindices[x-1, y, z]] += 1
                    L[li, li] -= 1
                end
                if x != n
                    L[li, lindices[x+1, y, z]] += 1
                    L[li, li] -= 1
                end
                if y != 1
                    L[li, lindices[x, y-1, z]] += 1
                    L[li, li] -= 1
                end
                if y != m
                    L[li, lindices[x, y+1, z]] += 1
                    L[li, li] -= 1
                end
                if z != 1
                    L[li, lindices[x, y, z-1]] += 1
                    L[li, li] -= 1
                end
                if z != k
                    L[li, lindices[x, y, z+1]] += 1
                    L[li, li] -= 1
                end
            end
        end
    end

    L
end
