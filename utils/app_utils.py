def create_vocabulary():
    """
    Create the vocabulary used by one-hot vectors.
    Returns
    -------
    array-like
        vocabulary
    """
    vocab = ["correct",
             "wrong",
             "A_H_F_CT",
             "A_H_F_OT",
             "A_H_F_CO",
             "C_H_F_CT",
             "C_H_F_OT",
             "C_H_F_CO",
             "G_H_F_CT",
             "G_H_F_OT",
             "G_H_F_CO",
             "A_H_M_CT",
             "A_H_M_OT",
             "A_H_M_CO",
             "C_H_M_CT",
             "C_H_M_OT",
             "C_H_M_CO",
             "G_H_M_CT",
             "G_H_M_OT",
             "G_H_M_CO"
             ]
    return vocab
