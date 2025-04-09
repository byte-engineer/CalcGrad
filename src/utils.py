from PIL import Image





def is_same_dim(lst1, lst2) -> bool:
    if not isinstance(lst1, list) or not isinstance(lst2, list):
        return True

    if len(lst1) != len(lst2):
        return False

    for sublist1, sublist2 in zip(lst1, lst2):
        if not is_same_dim(sublist1, sublist2):
            return False
    return True


def shape(_list: list) -> tuple:
    if not isinstance(list, _list):
        return ()
    return (len(_list),) + shape(_list[0])



from graphviz import Digraph


def draw_graph(root, dispdata=True):
    dot = Digraph(format="svg", graph_attr={"rankdir": "LR", "bgcolor": "lightgray"})  # Set background color

    nodes, edges = set(), set()

    def build_graph(root_value, dispdata):
        if root_value not in nodes:
            nodes.add(root_value)
            # Create a node for the value with white text on a blue background
            dot.node(str(id(root_value)), f"{round(root_value.data, 5) if dispdata else round(root_value.grad, 5)}", 
                     shape="record", style="rounded,filled", fillcolor="lightblue", fontcolor="black")

            # Create a separate operation node with shape="none", small font, and minimal size
            if root_value.op:
                op_node_id = f"op_{id(root_value)}"
                dot.node(op_node_id, root_value.op, shape="none", fontsize="20", width="0.1", height="0.1", fontcolor="black")
                dot.edge(op_node_id, str(id(root_value)))  # Connect op node to value node

            for child in root_value.children:
                edges.add((child, root_value))
                build_graph(child, dispdata)

    build_graph(root, dispdata)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), f"op_{id(n2)}")  # Connect value nodes to operation nodes

    return dot


def view_dot(dot):
    dot.render('out/graph', format='png')  # Save the graph
    image = Image.open('out/graph.png')
    image.show()
