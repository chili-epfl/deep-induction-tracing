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
             'C_E_F_T',
             'C_E_F_C',
             'C_E_F_O',
             'A_E_F_T',
             'A_E_F_O',
             'A_E_F_C',
             'G_E_F_C',
             'G_E_F_T',
             'G_E_F_O',
             'A_E_M_T',
             'A_E_M_O',
             'A_E_M_C',
             'G_E_M_O',
             'G_E_M_C',
             'G_E_M_T',
             'C_E_M_O',
             'C_E_M_C',
             'C_E_M_T',
             'C_H_F_CO',
             'C_H_F_CT',
             'C_H_F_OT',
             'G_H_F_OT',
             'G_H_F_CO',
             'G_H_F_CT',
             'A_H_F_CT',
             'A_H_F_OT',
             'A_H_F_CO',
             'C_H_M_CO',
             'C_H_M_CT',
             'C_H_M_OT',
             'A_H_M_CT',
             'A_H_M_OT',
             'A_H_M_CO',
             'G_H_M_OT',
             'G_H_M_CO',
             'G_H_M_CT']
    return vocab
