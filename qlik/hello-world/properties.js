define([], function () {
  'use strict';

  var custom = {
    label: 'Custom',
    ref: 'props.custom',
    layoutRef: 'custom',
    type: 'string',
    component: 'string',
    defaultValue: 'TEMP VALUE',
  }

  var appearance = {
    uses: 'settings',
  };

  return {
    key: 'cbk',
    type: 'items',
    component: 'accordion',
    items: {
      appearance: appearance,
      custom: custom,
    }
  };
});