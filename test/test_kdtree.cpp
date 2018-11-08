/* test_kdtree.cpp
 *
 *     Author: Fabian Meyer
 * Created On: 08 Nov 2018
 */

#include <catch.hpp>
#include <kdtree_eigen.h>

using namespace kdt;

TEST_CASE("KDTree")
{
    KDTreed kdtree;

    SECTION("build")
    {
        Eigen::MatrixXd data(3, 3);
        data << 1, 3, 0,
            0, 1, 2,
            3, 2, 0;
        kdtree.setBucketSize(2);
        kdtree.setData(data);
        kdtree.build();

        REQUIRE(kdtree.tree() != nullptr);
        REQUIRE(kdtree.tree()->left != nullptr);
        REQUIRE(kdtree.tree()->left->isLeaf());
        REQUIRE(kdtree.tree()->left->idx.size() == 2);
        REQUIRE(kdtree.tree()->left->idx(0) == 0);
        REQUIRE(kdtree.tree()->left->idx(1) == 2);

        REQUIRE(kdtree.tree()->right != nullptr);
        REQUIRE(kdtree.tree()->right->isLeaf());
        REQUIRE(kdtree.tree()->right->idx.size() == 1);
        REQUIRE(kdtree.tree()->right->idx(0) == 1);
    }
}
