from scipy.stats import mannwhitneyu
import numpy as np


def read_data(folder):
    l = []
    f = open(folder + "data_" + str(GENERATIONS) + ".txt", "r")

    while True:
        line = f.readline()
        if not line:
            break
        l.append(float(line))

    return l

def effect_size_wx(stat,n, n_ob):
    """
    n: size of effective sample (zero differences are excluded!)
    n_ob: number of observations = size sample 1 + size sample 2
    """
    mean = n*(n+1)/4
    std = np.sqrt(n*(n+1)*(2*n+1)/24)
    z_score = (stat - mean)/std
    return z_score/np.sqrt(n_ob)


def effect_size_mw(stat,n1,n2):
    """
    n1: size of the first sample
    n2: size of the second sample
    n_ob: number of observations
    """
    n_ob = n1 + n2 
    mean = n1*n2/2
    std = np.sqrt(n1*n2*(n1+n2+1)/12)
    z_score = (stat - mean)/std
    # print(z_score)
    return z_score,z_score/np.sqrt(n_ob)


def compare(algo1, algo2):
    # compare samples
    stat, p = mannwhitneyu(algo1, algo2)
    # bonferroni
    p = p*2
    print('Statistics=%.3f, p=%.3f' % (stat, p))
    # interpret
    alpha = 0.05
    if p > alpha:
        print('Same distribution (fail to reject H0)')
    else:
        print('Different distribution (reject H0)')

    media_a = np.mean(algo1)
    media_b = np.mean(algo2)

    z_score, r = effect_size_mw(stat, len(algo1), len(algo2))

    print("Effect Size: %f" % r)
    # box_plot([aa,bb], ["A","B"])
    # if abs(r) <= 0.1:
    #     print("~")
    if abs(r) <= 0.3:
        if media_a < media_b:
            print("-")
        else:
            print("+")
    elif abs(r) > 0.3 and abs(r) <= 0.5:
        if media_a < media_b:
            print("- -")
        else:
            print("+ +")
    elif abs(r) > 0.5 and abs(r) <= 1:
        if media_a < media_b:
            print("- - -")
        else:
            print("+ + +")
    else:
        print("error")
    # if abs(r) <= 0.1:
    #     print("~")
    # elif abs(r) > 0.1 and abs(r) <= 0.3:
    #     if media_a < media_b:
    #         print("-")
    #     else:
    #         print("+")
    # elif abs(r) > 0.3 and abs(r) <= 0.5:
    #     if media_a < media_b:
    #         print("- -")
    #     else:
    #         print("+ +")
    # elif abs(r) > 0.5 and abs(r) <= 1:
    #     if media_a < media_b:
    #         print("- - -")
    #     else:
    #         print("+ + +")
    # else:
    #     print("error")

    print()
    
GENERATIONS = 200


if __name__ == "__main__":
    path = "testes_finais_tese/"
    # problems = ["quad", "pagie", "bostonhousing", "5parity", "11mult", "ant"]
    problems = ["quad", "pagie", "bostonhousing"]

    std_psge_std = read_data("/home/jessica/mut_level/psge/sge/mutation_level_pagie/standard_mut10/1.0/")
    std_psge_mut_level = read_data("/home/jessica/mut_level/psge/sge/mutation_level_pagie/prob_mut_1.0_gauss_sd_0.0025/1.0/")
    psge_std = read_data("/home/jessica/mut_level/psge/sge/mutation_level_pagie_extended/standard_mut10/1.0/")
    psge_mut_level = read_data( "/home/jessica/mut_level/psge/sge/mutation_level_pagie_extended/prob_mut_1.0_gauss_sd_0.0025/1.0/")
    
    #std_psge_std = read_data("/home/jessica/mut_level/psge/sge/mutation_level_bh/standard_200gen_mut10/1.0/")
    #std_psge_mut_level = read_data("/home/jessica/mut_level/psge/sge/mutation_level_bh/prob_mut_1.0_gauss_sd_0.001/1.0/")
    #psge_std = read_data("/home/jessica/mut_level/psge/sge/mutation_level_bh_extended/standard_mut10/1.0/")
    #psge_mut_level = read_data( "/home/jessica/mut_level/psge/sge/mutation_level_bh_extended/prob_mut_1.0_gauss_sd_0.001/1.0/")
    

    print("SM+SG vs FM+SG")
    compare(std_psge_std, std_psge_mut_level)
    print("SM+EG vs FM+EG")
    compare(psge_std, psge_mut_level)
    print("SM+SG vs SM+EG")
    compare(std_psge_std, psge_std)    
    print("FM+SG vs FM+EG")
    compare(std_psge_mut_level, psge_mut_level)

    
    # for problem in problems:
    #     print("----------",problem,"--------------")
    #     ge = read_data(path + "ge/" + problem, "/data_50.txt")
    #     pge = read_data(path + "m1/" + problem + "/1.0", "/data_50.txt")
    #     # copge = read_data(path + "m2_1mut_por_nt/" + problem + "/5.0/0.50", "/data_50.txt")
    #     sge = read_data(path + "sge/" + problem, "/data_50.txt")
    #     psge = read_data(path + "psge/" + problem + "/1.0", "/data_50.txt")
    #     psge_delay = read_data("/home/jessicamegane/Documents/temp_psge_delay/test_psge_fix/" + problem + "/1.0/15", "/data_50.txt")
    #     copsge = read_data(path + "copsge/" + problem + "/5.0/0.5", "/data_50.txt")
    #     copsge_delay = read_data("/home/jessicamegane/Documents/temp_copsge_delay/test_copsge_fix/" + problem + "/5.0/0.5/15", "/data_50.txt")

    #     print("Co-PSGE_DELAY - Co-PSGE")
    #     compare(copsge, copsge_delay)
    #     # print("PGE - GE",)
    #     # compare(ge, pge)

    #     # print("PGE - SGE")
    #     # compare(sge, pge)

    #     # print("Co-PGE - GE",)
    #     # compare(ge, copge)

    #     # print("Co-PGE - SGE")
    #     # compare(sge, copge)

    #     # print("PSGE - GE")
    #     # compare(ge, psge)

    #     # print("PSGE - PGE")
    #     # compare(pge, psge)

    #     # print("PSGE - SGE")
    #     # compare(sge, psge)


    #     # print("Co-PSGE - GE")
    #     # compare(ge, copsge)

    #     # print("Co-PSGE - PGE")
    #     # compare(pge, copsge)

    #     # print("Co-PSGE - SGE")
    #     # compare(sge, copsge)


    #     if problem == "bostonhousing":
    #         print("----------",problem," Test --------------")
    #         ge = read_data(path + "ge/" + problem, "/_testdata_50.txt")
    #         pge = read_data(path + "m1/" + problem + "/1.0", "/_testdata_50.txt")
    #         copge = read_data(path + "m2_1mut_por_nt/" + problem + "/5.0/0.50", "/_testdata_50.txt")
    #         sge = read_data(path + "sge/" + problem, "/_testdata_50.txt")
    #         psge = read_data(path + "psge/" + problem + "/1.0", "/_testdata_50.txt")
    #         psge_delay = read_data("/home/jessicamegane/Documents/temp_psge_delay/test_psge_fix/" + problem + "/1.0/15", "/_testdata_50.txt")
    #         copsge = read_data(path + "copsge/" + problem + "/5.0/0.5", "/_testdata_50.txt")
    #         copsge_delay = read_data("/home/jessicamegane/Documents/temp_copsge_delay/test_copsge_fix/" + problem + "/5.0/0.5/15", "/_testdata_50.txt")

    #         print("Co-PSGE_DELAY - CO-PSGE")
    #         compare(copsge, copsge_delay)
    #         # print("PGE - GE",)
    #         # compare(ge, pge)

    #         # print("PGE - SGE")
    #         # compare(sge, pge)

    #         # print("Co-PGE - GE",)
    #         # compare(ge, copge)

    #         # print("Co-PGE - SGE")
    #         # compare(sge, copge)

    #         # print("PSGE - GE")
    #         # compare(ge, psge)

    #         # print("PSGE - PGE")
    #         # compare(pge, psge)

    #         # print("PSGE - SGE")
    #         # compare(sge, psge)

    #         # print("Co-PSGE - GE",)
    #         # compare(ge, copsge)

    #         # print("Co-PSGE - SGE")
    #         # compare(sge, copsge)

    #         # print("Co-PSGE - PGE")
    #         # compare(pge, copsge)
