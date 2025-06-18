#pragma once
#include <vector>

/// @brief  3D array class
/// @tparam T Type of the array elements
template <typename T>
class Array3D {
    private:
        int i_total, j_total, k_total;
        std::vector<T> array;
    public:
        /// @brief Constructor
        /// @param i_total_ Total size in x direction including margin
        /// @param j_total_ Total size in y direction including margin
        /// @param k_total_ Total size in z direction including margin
        Array3D(int i_total_, int j_total_, int k_total_):
            i_total(i_total_),
            j_total(j_total_),
            k_total(k_total_),
            array(i_total_ * j_total_ * k_total_) {}

        /// @brief overload function for accessing the 3D array elements
        /// @param i i index
        /// @param j j index
        /// @param k k index
        /// @return Reference to the element at (i, j, k)
        T& operator()(int i, int j, int k) {
            assert(i >= 0 && i < i_total);
            assert(j >= 0 && j < j_total);
            assert(k >= 0 && k < k_total);
            return array[i * j_total * k_total + j * k_total + k];
        }
        const T& operator()(int i, int j, int k) const {
            assert(i >= 0 && i < i_total);
            assert(j >= 0 && j < j_total);
            assert(k >= 0 && k < k_total);
            return array[i * j_total * k_total + j * k_total + k];
        }

        T* data() { return array.data(); }
        const T* data() const { return array.data(); }

        int size_x () const { return i_total; }
        int size_y () const { return j_total; }
        int size_z () const { return k_total; }
        int size () const { return i_total * j_total * k_total; }

        void copy_from(const Array3D& other) {
            assert(i_total == other.i_total);
            assert(j_total == other.j_total);
            assert(k_total == other.k_total);
            std::copy(other.array.begin(), other.array.end(), array.begin());
        }
};