import copy

import logging
logger = logging.getLogger(__name__)

def update_dict(original, new):
    """
    Update nested dictionary (dictionary possibly containing dictionaries)
    If a field is present in new and original, take the value from new.
    If a field is present in new but not original, insert this field 
    :param original: source dictionary
    :type original: dict
    :param new: dictionary to take new values from
    :type new: dict
    :return: updated dictionary
    :rtype: dict
    """

    updated = copy.deepcopy(original)

    for key, value in original.items():
        if key in new.keys():
            if isinstance(value, dict):
                updated[key] = update_dict(value, new[key])
            else:
                updated[key] = new[key]

    return updated
