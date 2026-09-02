from collections import defaultdict


def find_solved_node(submission):
    """
    Finds the node in a submission with "solved" in the reply
    """
    reply_stack = submission["replies"][::]
    author_id = submission["author"]
    solved_nodes = set()
    while len(reply_stack) > 0:
        reply = reply_stack.pop(0)
        if "solved" in reply["body"].lower() and reply.get("author_name") == author_id:
            solved_nodes.add(reply["id"])

        if reply["replies"] is not None:
            reply_stack.extend(reply["replies"])

    return solved_nodes


def _gather_desc(reply):
    """
    Helper function: Gathers descendants (list of IDs of all descendants)
    """
    if reply["replies"] is None:
        reply["descendants"] = list()
        return list()
    desc = list()
    for r in reply["replies"]:
        assert r["id"] not in desc
        desc.append(r["id"])
        desc.extend(_gather_desc(r))
    reply["descendants"] = desc
    return desc


def gather_descendants(submission):
    """
    Gathers descendants (list of IDs of all descendants)
    """
    # create a new property in 'descendants', containing all descendant ids
    for reply in submission["replies"]:
        desc = _gather_desc(reply)
        reply["descendants"] = list(desc)


def get_formatted_conv_reply(reply, op_id):
    """
    Reply -> dictionary only with certain information
    """
    ut = {
        "utterance": reply["body"],
        "author": reply["author_name"] if "author_name" in reply else None,
        "id": reply["id"],
    }

    ut["is_op"] = ut["author"] == op_id

    return ut


def find_path_to_node(submission, match_id):
    """
    Given a submission and a matching id, this function
    first navigates to the matching ID, and then navigates back to the
    top, effectively extracting a single path from the root to the matching node.
    """
    match_reply = None
    for reply in submission["replies"]:
        if match_id in reply["descendants"]:
            match_reply = reply

    if match_reply is None:
        return None, None

    # original poster's id
    op_id = submission["author"]
    path = [match_reply["id"]]
    convo = [get_formatted_conv_reply(match_reply, op_id)]
    reply_stack = match_reply["replies"][::]

    while len(reply_stack) > 0:
        reply = reply_stack.pop(0)

        if reply["id"] == match_id:
            path.append(reply["id"])
            convo.append(get_formatted_conv_reply(reply, op_id))

        if match_id in reply["descendants"]:
            path.append(reply["id"])
            convo.append(get_formatted_conv_reply(reply, op_id))

        if reply["replies"] is not None:
            reply_stack.extend(reply["replies"])

    return path, convo


def create_conv_dicts(submissions):
    """
    Finds all solved paths. 

    Returns (conv_dictionary, solved_node_counts, len_convos)

    conv_dictionary -> solved path (incl. submission)

    solved_node_counts -> # of solved nodes -> count

    len_convos -> array of lengths of solved paths 

    """

    counts = defaultdict(int)
    convos = {}
    len_convos = []
    not_found = 0
    for submission in submissions:
        solved_nodes = find_solved_node(submission)
        counts[len(solved_nodes)] += 1

        if len(solved_nodes) == 1:
            match_id = list(solved_nodes)[0]
            gather_descendants(submission)

            path, c = find_path_to_node(submission, match_id)
            if path is None:
                not_found += 1
                continue

            convos[submission["id"]] = {
                "submission": submission,
                "solved_path": c,
                "solved_path_ids": path
            }
            len_convos.append(len(path))

    return convos, counts, len_convos
