#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include <miso/mhd_model_base.hpp>

// Source terms are optional: a term that the Source type does not define is
// treated as 0 (source_ro, ..., source_ei in mhd_integrator.hpp).

using namespace miso;
using Real = double;

namespace {

/// @brief Defines only vx and bz
struct PartialSource {
  Real vx(mhd::FieldsView<const Real>, int i, int, int) const { return i + 1; }
  Real bz(mhd::FieldsView<const Real>, int, int, int k) const { return -k; }
};

/// @brief Defines every term
struct FullSource {
  Real ro(mhd::FieldsView<const Real>, int, int, int) const { return 1; }
  Real vx(mhd::FieldsView<const Real>, int, int, int) const { return 2; }
  Real vy(mhd::FieldsView<const Real>, int, int, int) const { return 3; }
  Real vz(mhd::FieldsView<const Real>, int, int, int) const { return 4; }
  Real bx(mhd::FieldsView<const Real>, int, int, int) const { return 5; }
  Real by(mhd::FieldsView<const Real>, int, int, int) const { return 6; }
  Real bz(mhd::FieldsView<const Real>, int, int, int) const { return 7; }
  Real ei(mhd::FieldsView<const Real>, int, int, int) const { return 8; }
  Real dt_limit() const { return 0.1; }
};

/// @brief vx takes FieldsView<Real> (not const): a typo that must not be
/// silently ignored
struct MalformedSource {
  Real vx(mhd::FieldsView<Real>, int, int, int) const { return 1; }
};

/// @brief dt_limit is not const
struct MalformedDtLimit {
  Real dt_limit() { return 1; }
};

// A member with the name but a wrong signature is detected as present but not
// callable, which source_vx and ModelBase::update reject by static_assert.
static_assert(mhd::impl_source::has_member_vx<MalformedSource>::value);
static_assert(!mhd::impl_source::has_vx<MalformedSource, Real>::value);
static_assert(mhd::has_member_dt_limit<MalformedDtLimit>::value);
static_assert(!mhd::has_dt_limit<MalformedDtLimit>::value);
// A missing member is simply absent.
static_assert(
    !mhd::impl_source::has_member_vx<mhd::EmptySourceTerm<Real>>::value);

}  // namespace

TEST_CASE("Source terms are optional" * doctest::test_suite("mhd_source")) {
  mhd::Fields<Real, backend::Host> qq(4, 4, 4);
  const auto q = qq.const_view();

  SUBCASE("EmptySourceTerm defines no term") {
    const mhd::EmptySourceTerm<Real> src;
    CHECK(mhd::source_ro(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_vx(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_vy(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_vz(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_bx(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_by(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_bz(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_ei(src, q, 1, 2, 3) == 0);
    CHECK_FALSE(mhd::has_dt_limit<mhd::EmptySourceTerm<Real>>::value);
  }

  SUBCASE("Only the defined terms are used") {
    const PartialSource src;
    CHECK(mhd::source_vx(src, q, 1, 2, 3) == 2);
    CHECK(mhd::source_bz(src, q, 1, 2, 3) == -3);
    CHECK(mhd::source_ro(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_vy(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_vz(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_bx(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_by(src, q, 1, 2, 3) == 0);
    CHECK(mhd::source_ei(src, q, 1, 2, 3) == 0);
    CHECK_FALSE(mhd::has_dt_limit<PartialSource>::value);
  }

  SUBCASE("All terms are used when defined") {
    const FullSource src;
    CHECK(mhd::source_ro(src, q, 0, 0, 0) == 1);
    CHECK(mhd::source_vx(src, q, 0, 0, 0) == 2);
    CHECK(mhd::source_vy(src, q, 0, 0, 0) == 3);
    CHECK(mhd::source_vz(src, q, 0, 0, 0) == 4);
    CHECK(mhd::source_bx(src, q, 0, 0, 0) == 5);
    CHECK(mhd::source_by(src, q, 0, 0, 0) == 6);
    CHECK(mhd::source_bz(src, q, 0, 0, 0) == 7);
    CHECK(mhd::source_ei(src, q, 0, 0, 0) == 8);
    CHECK(mhd::has_dt_limit<FullSource>::value);
  }
}
