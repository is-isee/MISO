#pragma once
#include <vector>

/// @brief  4D array class
/// @tparam T Type of the array elements
template <typename T>
class Array4D {
    private:
        int i_total, j_total, k_total, l_total;
        std::vector<T> array;
    public:
        /// @brief Constructor
        /// @param i_total_ Total size in x direction including margin
        /// @param j_total_ Total size in y direction including margin
        /// @param k_total_ Total size in z direction including margin
        /// @param l_total_ Total size in w direction 
        Array4D(int i_total_, int j_total_, int k_total_, int l_total_):
            i_total(i_total_),
            j_total(j_total_),
            k_total(k_total_),
            l_total(l_total_),
            array(i_total_ * j_total_ * k_total_ * l_total_) {}

        /// @brief overload function for accessing the 3D array elements
        /// @param i i index
        /// @param j j index
        /// @param k k index
        /// @param l l index
        /// @return Reference to the element at (i, j, k, l)
        T& operator()(int i, int j, int k, int l) {
            return array[i * j_total * k_total * l_total + j * k_total * l_total + k * l_total + l];
        }
        const T& operator()(int i, int j, int k, int l) const {
            return array[i * j_total * k_total * l_total + j * k_total * l_total + k * l_total + l];
        }

        T* data() { return array.data(); }
        const T* data() const { return array.data(); }

        int size_x () const { return i_total; }
        int size_y () const { return j_total; }
        int size_z () const { return k_total; }
        int size_w () const { return l_total; }
        int size () const { return i_total * j_total * k_total * l_total; }

        void copy_from(const Array4D& other) {
            std::copy(other.array.begin(), other.array.end(), array.begin());
        }
};