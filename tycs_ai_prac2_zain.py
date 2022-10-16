from RMP import dict_gn

start = 'Arad'
goal = 'Bucharest'
result = ''

def DLS(city, visited_stack, start_limit, end_limit):
    global result
    found = 0
    result = result + city + ' '
    visited_stack.append(city)
    if city == goal: 
        return 1
    if start_limit == end_limit:
        return 0
    for eachcity in dict_gn[city].keys():
        if eachcity not in visited_stack:
            found = DLS(eachcity, visited_stack, start_limit + 1, end_limit)
            if found:
                return found

def IDDFS(city, visited_stack, end_limit):
    global result
    for i in range(0, end_limit):
        print("Searching at limit: ", i)
        found = DLS(city, visited_stack, 0, i)
        if found:
            print("Found ")
            break
        else:
            print("Not found")
            print(result)
            print('--------')
            visited_stack = []
            result = ''

def main():
    visited_stack = []
    IDDFS(start, visited_stack, 9)
    print('IDDFS Traversal from ', start, 'to ', goal, 'is: ', result)

main()
