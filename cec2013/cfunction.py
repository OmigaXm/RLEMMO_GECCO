###############################################################################
# Version: 1.1
# Last modified on: 3 April, 2016
# Developers: Michael G. Epitropakis
#      email: m_(DOT)_epitropakis_(AT)_lancaster_(DOT)_ac_(DOT)_uk
###############################################################################
import numpy as np

# UNCOMMENT APPROPRIATELY
# MINMAX = 1		# Minimization
MINMAX = -1  # Maximization


class CFunction(object):
    __dim_ = -1
    __nofunc_ = -1
    __C_ = 2000.0
    __lambda_ = None
    __sigma_ = None
    __bias_ = None
    __O_ = None
    __M_ = None
    __weight_ = None
    __lbound_ = None
    __ubound_ = None
    __fi_ = None
    __z_ = None
    __f_bias_ = 0
    __fmaxi_ = None
    __tmpx_ = None
    __function_ = None

    def __init__(self, dim, nofunc):
        self.__dim_ = dim
        self.__nofunc_ = nofunc

    def evaluate(self, x):
        pass

    def get_lbound(self, ivar):
        assert ivar >= 0 and ivar < self.__dim_, [
            "ivar is not in valid variable range: %d not in [0,%d]" % ivar,
            self.__dim_,
        ]
        return self.__lbound_[ivar]

    def get_ubound(self, ivar):
        assert ivar >= 0 and ivar < self.__dim_, [
            "ivar is not in valid variable range: %d not in [0,%d]" % ivar,
            self.__dim_,
        ]
        return self.__ubound_[ivar]

    def __evaluate_inner_(self, x):
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")
        self.__fi_ = np.zeros((x.shape[0], self.__nofunc_))

        self.__calculate_weights(x)
        for i in range(self.__nofunc_):
            self.__transform_to_z(x, i)
            self.__fi_[:, i] = self.__function_[i](self.__z_)

        tmpsum = np.zeros((x.shape[0], self.__nofunc_))
        for i in range(self.__nofunc_):
            tmpsum[:, i] = self.__weight_[:, i] * (
                self.__C_ * self.__fi_[:, i] / self.__fmaxi_[i] + self.__bias_[i]
            )
        return np.sum(tmpsum, -1) * MINMAX + self.__f_bias_

    def __calculate_weights(self, x):
        self.__weight_ = np.zeros((x.shape[0],self.__nofunc_))
        
        for i in range(self.__nofunc_):
            mysum = np.sum((x - self.__O_[i]) ** 2, -1)
            self.__weight_[:,i] = np.exp(
                -mysum / (2.0 * self.__dim_ * self.__sigma_[i] * self.__sigma_[i])
            )
        
        maxw = np.max(self.__weight_, -1)
        # maxi = self.__weight_.argmax(axis=0)
        maxw10 = maxw ** 10

        seek = np.where(self.__weight_ != np.repeat(maxw, self.__nofunc_, 0).reshape(x.shape[0], self.__nofunc_))
        self.__weight_[seek] = (self.__weight_ * (1.0 - np.repeat(maxw10, self.__nofunc_,0).reshape(x.shape[0], self.__nofunc_)))[seek]

        mysum = np.sum(self.__weight_, -1)
        seek0 = np.where(mysum == 0)[0]
        seekn0 = np.where(mysum != 0)[0]
        self.__weight_[seek0,:] = 1.0 / (1.0 * self.__nofunc_)
        self.__weight_[seekn0,:] = self.__weight_[seekn0,:] / mysum[seekn0][:,None]

        

    def __calculate_fmaxi(self):
        self.__fmaxi_ = np.zeros(self.__nofunc_)
        if self.__function_ == None:
            raise NameError("Composition functions' dict is uninitialized")

        x5 = 5 * np.ones(self.__dim_)

        for i in range(self.__nofunc_):
            self.__transform_to_z_noshift(x5[None, :], i)
            self.__fmaxi_[i] = self.__function_[i](self.__z_)[0]

    def __transform_to_z_noshift(self, x, index):
        # z_i = (x)/\lambda_i
        tmpx = (x / self.__lambda_[index])
        # Multiply z_i * M_i
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __transform_to_z(self, x, index):
        # Calculate z_i = (x - o_i)/\lambda_i
        tmpx = (x - self.__O_[index]) / self.__lambda_[index]
        # Multiply z_i * M_i
        self.__z_ = np.dot(tmpx, self.__M_[index])

    def __load_rotmat(self, fname):
        self.__M_ = []

        with open(fname, "r") as f:
            tmp = np.zeros((self.__dim_, self.__dim_))
            cline = 0
            ctmp = 0
            for line in f:
                line = line.split()
                if line:
                    line = [float(i) for i in line]
                    # re initialize array when reached dim
                    if ctmp % self.__dim_ == 0:
                        tmp = np.zeros((self.__dim_, self.__dim_))
                        ctmp = 0

                    # add line to tmp
                    tmp[ctmp] = line[: self.__dim_]
                    # if we loaded self.__nofunc_ * self.__dim_-1 lines break
                    if cline >= self.__nofunc_ * self.__dim_ - 1:
                        break
                    # add array to M_ when it is fully created
                    if cline % self.__dim_ == 0:
                        self.__M_.append(tmp)
                    ctmp = ctmp + 1
                    cline = cline + 1


# Sphere function
def FSphere(x):
    return np.sum(x ** 2, -1)


# Rastrigin's function
def FRastrigin(x):
    return np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10, -1)


# Griewank's function
def FGrienwank(x):
    dim = x.shape[1]
    s = np.sum(x ** 2, -1)
    p = np.ones(x.shape[0])
    for i in range(dim):
        p *= np.cos(x[:, i] / np.sqrt(1 + i))
    return 1 + s / 4000 - p


# Weierstrass's function
def FWeierstrass(x):
    dim = x.shape[1]
    a, b, k_max = 0.5, 3.0, 20
    sum1, sum2 = 0, 0
    for k in range(k_max + 1):
        sum1 += np.sum(np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * (x + 0.5)), -1)
        sum2 += np.power(a, k) * np.cos(2 * np.pi * np.power(b, k) * 0.5)
    return sum1 - dim * sum2


# FEF8F2 function
def FEF8F2(x):
    z = x + 1
    z_ = np.concatenate((z[:, 1:], z[:, :1]), -1)
    _z = z
    tmp1 = _z ** 2 - z_
    temp = 100 * tmp1 * tmp1 + (_z - 1) ** 2
    res = np.sum(temp * temp / 4000 - np.cos(temp) + 1, -1)
    return res

